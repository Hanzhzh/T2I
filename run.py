import argparse
import datetime
import json
import math
import os
import random
import numpy as np
import time
import torch
from ruamel.yaml import YAML
import logging
from pathlib import Path

from optim import create_optimizer
from scheduler import create_scheduler
import utils

from model.tokenization_bert import BertTokenizer
from model.tokenization_roberta import RobertaTokenizer

# 自定义数据集
from dataset.dataloader import create_dataset, create_loader, custom_collate_fn

from model.bench import Bench
from train_evaluate import train_v2, train_v3, evaluation, evaluation_simple_amp, evaluation_simple_amp_optimized, train_v1


# 设置随机种子
def setup_seed(seed: int = 42) -> None:
    print(f"seed的值为{seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True  # 开启cuDNN基准模式，提高性能
    print(f"随机种子设置为: {seed}")

def load_config(config_path):
    """
    加载 YAML 配置文件
    """
    yaml = YAML(typ='rt')
    with open(config_path, 'r') as f:
        return yaml.load(f)

def parse_gpus(gpu_str):
    """
    解析GPU字符串，返回GPU列表和GPU数量。
    """
    gpus = [int(x) for x in gpu_str.split(',') if x.strip().isdigit()]
    return gpus, len(gpus)

def main(args, config):
    """
    主函数
    """
    setup_seed()  # 设置种子
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = f'./output/{current_time}'

    # 解析GPU配置
    gpus, num_gpus = parse_gpus(config['GPUS'])
    print(f"解析到的GPU列表: {gpus}, GPU数量: {num_gpus}")

    # 设置使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 设置环境变量以指定GPU设备（可选，根据需要保留或移除）
    os.environ["CUDA_VISIBLE_DEVICES"] = config['GPUS']

    print("创建模型", flush=True)
    model = Bench(config)  # 创建模型实例
    if args.checkpoint:
        model.load_pretrained(args.checkpoint, config, is_eval=False)  # 加载预训练权重
        print(f"已加载预训练权重来自 {args.checkpoint}")
    else:
        print("从零开始训练：未加载预训练权重")

    # model.load_pretrained(args.checkpoint, config, is_eval=False)  # 加载预训练权重
    # if args.evaluate:
    #     model = model.half()  # 如果是评估模式，转换为半精度
    model = model.to(device)  # 将模型转移到设备
    print("### 模型参数总数为: ", sum(p.numel() for p in model.parameters() if p.requires_grad))  # 打印模型参数总数

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    print("优化器初始化完成")

    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
        print("并行训练模式已启用")
    else:
        print("单GPU训练模式已启用")

    # 根据配置选择使用的分词器
    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("构建数据集", flush=True)
    train_dataset, test_dataset = create_dataset('I2T_university', config, args.evaluate)  # 创建训练和测试数据集

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:

        # ====== 训练结束后：手动加载最新或最优模型，再评估 ======
        ck_path = "./output/train"
        last_checkpoint_path = os.path.join(ck_path, f'')
        checkpoint_data = torch.load(last_checkpoint_path, map_location='cpu')
        model_state = checkpoint_data['model']
        if num_gpus > 1:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        print("Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[0],
                                    is_trains=[False],
                                    collate_fns=[custom_collate_fn])[0]
        # 进行评估
        score_dict = evaluation_simple_amp_optimized(model, test_loader, tokenizer, device, config)
        print(f"评估的得分字典为{score_dict}")

    else:
        # 训练模式
        print("开始训练", flush=True)

        train_dataset_size = len(train_dataset)

        # 创建采样器（无需分布式采样器）
        samplers = [None, None]

        train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers=samplers,
                                                  batch_size=[config['batch_size_train'],
                                                      config['batch_size_test']],
                                                  num_workers=[0, 0],
                                                  is_trains=[True, False],
                                                  collate_fns=[custom_collate_fn, custom_collate_fn])

        # 创建优化器和学习率调度器

        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * max(1, num_gpus)))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        print("学习率调度器初始化完成")

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            print(f"Epoch {epoch+1}/{max_epoch}")

            # 进行训练
            train_stats = train_v2(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            # 记录训练日志
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 保存检查点
            if epoch <= config['schedular']['epochs'] - 1:
                save_obj = {
                    'model': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch}.pth')
                torch.save(save_obj, checkpoint_path)
                logging.info(f"Checkpoint已保存至 {checkpoint_path}")

            torch.cuda.empty_cache()  # 清理CUDA缓存
        # ====== 训练结束后：手动加载最新或最优模型，再评估 ======
        last_checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{max_epoch - 1}.pth')
        checkpoint_data = torch.load(last_checkpoint_path, map_location='cpu')
        model_state = checkpoint_data['model']

        if num_gpus > 1:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        #     # 用于打印前20个参数
        # counter_1 = 0
        # for name, param in model.named_parameters():
        #     if param.data.dim() >= 2:
        #         print(f"加载权重之后： - {name}: {param.data[0][0]}")
        #     elif param.data.dim() == 1:
        #         print(f"加载权重之后： - {name}: {param.data[0]}")
        #     elif param.data.dim() == 0:
        #         print(f"加载权重之后： - {name}: {param.data.item()}")
        #     else:
        #         print(f"加载权重之后： - {name}: {param.data}")
        #
        #     counter_1 += 1
        #     if counter_1 >= 20:
        #         break  # 打印前20个参数后退出
        # # 如果想用半精度评估，可以再加一句:
        # # model.half()

        print("开始评估", flush=True)
        # 创建 test_loader (有些情况下已经有了, 这里示例中重复说明)
        test_loader = create_loader(
            [test_dataset],
            [None],
            batch_size=[config['batch_size_test']],
            num_workers=[0],
            is_trains=[False],
            collate_fns=[custom_collate_fn]
        )[0]

        score_dict = evaluation_simple_amp_optimized(model, test_loader, tokenizer, device, config)
        print(f"评估的得分字典为 {score_dict}")
        # 记录最佳epoch（需要在训练过程中更新best和best_epoch变量）
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        os.system(f"cat {args.output_dir}/log.txt")  # 输出日志

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='i2t.yaml', type=str, help="if not given, use default")
    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', default='./output/train', type=str, help='for fine-tuning, local path; '
                                                                          'for pre-training, local and HDFS are both allowed.')

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录

    yaml = YAML()
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  # 保存配置文件到输出目录

    main(args, config)  # 调用主函数
