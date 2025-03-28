import torch
# from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
# the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys

sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler


class CLIP_Clean_Train():
    def __init__(self, rank, local_rank, args):
        """
        构造函数，初始化训练所需的各种组件，包括模型、优化器、日志记录器等。
        """
        # rank和local_rank用于分布式训练
        self.rank = rank
        self.local_rank = local_rank

        # 基础模型名称（例如'ViT-L/14'）
        self.base_model = args.base_model

        # 使用longclip.load_from_clip加载基础模型，并将其放到CPU
        self.model, _ = longclip.load_from_clip(
            self.base_model,
            device='cpu',
            download_root=args.download_root
        )
        # 设置模型为训练模式
        self.model.train()
        # 重写logit_scale，使之可训练，初始值由args.log_scale指定
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)

        # 将模型移动到GPU
        self.model = self.model.cuda()

        # 设置训练超参数
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length

        # 日志目录与权重保存目录的设置
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)

        # 用于TensorBoard可视化
        self.writer = SummaryWriter(self.logdir)

        # 把模型包装进DistributedDataParallel，分布式训练
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank]
        )

        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # AMP自动混合精度训练，减少显存占用
        self.scaler = GradScaler()

        # **新增：记录最佳测试集准确率**
        self.best_acc = 0.0

        # 初始化记录最佳模型路径的变量
        self.previous_best_ckpt_path = None

    # @torch.no_grad()
    # def test_epoch(self, dataloader):
    #     """
    #     在验证集/测试集上做一次epoch的测试，返回检索正确率。
    #     """
    #     rank = torch.distributed.get_rank()
    #     correct = 0
    #     total = 0

    #     for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):
    #         # 移动图像到GPU
    #         images = images.cuda()
    #         # 编码图像得到图像特征
    #         image_features = self.model.module.encode_image(images)
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #         # 对文本做tokenize并移动到GPU
    #         text = longclip.tokenize(text, truncate=True).cuda()
    #         # 编码文本得到文本特征
    #         text_feature = self.model.module.encode_text(text)
    #         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    #         # 计算准确率
    #         for i in range(text_feature.shape[0]):
    #             single_text = text_feature[i]
    #             sim = single_text @ image_features.T  # 相似度计算
    #             sim = sim.squeeze()
    #             correct_i = torch.argmax(sim)
    #             if i == correct_i:
    #                 correct += 1
    #             total += 1

    #     # 返回整批上的平均准确率
    #     if total == 0:
    #         return 0.0
    #     else:
    #         return correct / total

    # def test(self, epoch=0):
    #     """
    #     在验证集/测试集上进行一次完整的测试过程，并打印结果。
    #     """
    #     rank = torch.distributed.get_rank()
    #     if rank == 0:
    #         self.model.eval()
    #         # 构造验证集数据
    #         testset = share4v_val_dataset()
    #         testloader = torch.utils.data.DataLoader(
    #             testset,
    #             batch_size=100,
    #             num_workers=32,
    #             pin_memory=True
    #         )
    #         with torch.no_grad():
    #             acc = self.test_epoch(testloader)
    #             print("=====================================")
    #             print(f"Epoch {epoch} - test mean of share4v retrieval: {acc}")
    #             print("=====================================")

    #         # 测试完成后要把模型恢复到训练模式
    #         self.model.train()
    #         return acc
    #     else:
    #         # rank != 0时，不做测试
    #         return 0.0
    def test_epoch(self, dataloader):
        """
        在验证集/测试集上做一次epoch的测试，返回文本到图像和图像到文本的recall@1、recall@5、recall@10。
        """
        rank = torch.distributed.get_rank()
        correct_1_t2i = 0  # 文本到图像的 recall@1
        correct_5_t2i = 0  # 文本到图像的 recall@5
        correct_10_t2i = 0  # 文本到图像的 recall@10
        correct_1_i2t = 0  # 图像到文本的 recall@1
        correct_5_i2t = 0  # 图像到文本的 recall@5
        correct_10_i2t = 0  # 图像到文本的 recall@10
        total = 0

        for id, (images, roof, around_1, around_2) in enumerate(tqdm(dataloader, disable=(rank != 0))):
            # 移动图像到GPU
            images = images.cuda()

            # 编码图像得到图像特征
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 对文本做tokenize并移动到GPU
            roof = longclip.tokenize(roof, truncate=True).cuda()
            around_1 = longclip.tokenize(around_1, truncate=True).cuda()
            around_2 = longclip.tokenize(around_2, truncate=True).cuda()

            # 编码文本得到文本特征
            roof_feature = self.model.module.encode_text(roof)
            roof_feature = roof_feature / roof_feature.norm(dim=-1, keepdim=True)
            around_1_feature = self.model.module.encode_text(around_1)
            around_1_feature = around_1_feature / around_1_feature.norm(dim=-1, keepdim=True)
            around_2_feature = self.model.module.encode_text(around_2)
            around_2_feature = around_2_feature / around_2_feature.norm(dim=-1, keepdim=True)

            # 计算文本到图像的 recall@1、recall@5、recall@10
            for i in range(roof_feature.shape[0]):
                roof_text = roof_feature[i]
                around_1_text = around_1_feature[i]
                around_2_text = around_2_feature[i]

                # 计算与所有图像的相似度
                sim_r2i = roof_text @ image_features.T  # 文本到图像的相似度计算
                sim_r2i = sim_r2i.squeeze()
                sim_a12i = around_1_text @ image_features.T  # 文本到图像的相似度计算
                sim_a12i = sim_a12i.squeeze()
                sim_a22i = around_2_text @ image_features.T  # 文本到图像的相似度计算
                sim_a22i = sim_a22i.squeeze()

                sim_t2i = sim_r2i + sim_a12i + sim_a22i
                # 获取排序后的图像索引
                sorted_idx_t2i = torch.argsort(sim_t2i, descending=True)

                # 如果正确的图像在前K个中，则增加相应的recall计数
                if i in sorted_idx_t2i[:1]:
                    correct_1_t2i += 1
                if i in sorted_idx_t2i[:5]:
                    correct_5_t2i += 1
                if i in sorted_idx_t2i[:10]:
                    correct_10_t2i += 1

            # 计算图像到文本的 recall@1、recall@5、recall@10
            for i in range(images.shape[0]):
                single_image = image_features[i]

                # 计算与所有文本的相似度
                sim_i2r = single_image @ roof_feature.T  # 图像到文本的相似度计算
                sim_i2r = sim_i2r.squeeze()
                sim_i2a1 = single_image @ around_1_feature.T  # 图像到文本的相似度计算
                sim_i2a1 = sim_i2a1.squeeze()
                sim_i2a2 = single_image @ around_2_feature.T  # 图像到文本的相似度计算
                sim_i2a2 = sim_i2a2.squeeze()

                sim_i2t = sim_i2r + sim_i2a1 + sim_i2a2

                # 获取排序后的文本索引
                sorted_idx_i2t = torch.argsort(sim_i2t, descending=True)

                # 如果正确的文本在前K个中，则增加相应的recall计数
                if i in sorted_idx_i2t[:1]:
                    correct_1_i2t += 1
                if i in sorted_idx_i2t[:5]:
                    correct_5_i2t += 1
                if i in sorted_idx_i2t[:10]:
                    correct_10_i2t += 1

            total += images.shape[0]  # 统计总数

        # 计算各个recall的值
        recall_1_t2i = correct_1_t2i / total if total != 0 else 0.0
        recall_5_t2i = correct_5_t2i / total if total != 0 else 0.0
        recall_10_t2i = correct_10_t2i / total if total != 0 else 0.0
        recall_1_i2t = correct_1_i2t / total if total != 0 else 0.0
        recall_5_i2t = correct_5_i2t / total if total != 0 else 0.0
        recall_10_i2t = correct_10_i2t / total if total != 0 else 0.0

        return recall_1_t2i, recall_5_t2i, recall_10_t2i, recall_1_i2t, recall_5_i2t, recall_10_i2t

    def test(self, epoch=0):
        """
        在验证集/测试集上进行一次完整的测试过程，并打印结果。
        """
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.model.eval()
            # 构造验证集数据
            testset = share4v_val_dataset()  # 假设你有一个函数加载数据集
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=100,
                num_workers=32,
                pin_memory=True
            )
            with torch.no_grad():
                # 调用test_epoch计算recall
                recall_1_t2i, recall_5_t2i, recall_10_t2i, recall_1_i2t, recall_5_i2t, recall_10_i2t = self.test_epoch(testloader)
                
                # 打印结果
                print("=====================================")
                print(f"Epoch {epoch} - Text-to-Image recall@1: {recall_1_t2i:.4f}, recall@5: {recall_5_t2i:.4f}, recall@10: {recall_10_t2i:.4f}")
                print(f"Epoch {epoch} - Image-to-Text recall@1: {recall_1_i2t:.4f}, recall@5: {recall_5_i2t:.4f}, recall@10: {recall_10_i2t:.4f}")
                print("=====================================")

            # 测试完成后要把模型恢复到训练模式
            self.model.train()
            return recall_1_t2i, recall_5_t2i, recall_10_t2i, recall_1_i2t, recall_5_i2t, recall_10_i2t
        else:
            # rank != 0时，不做测试
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def train_epoch(self, dataloader, epoch, start_iter=0):
        """
        单个Epoch的训练流程。
        """
        # rank = torch.distributed.get_rank()
        # num_batches_per_epoch记录当前DataLoader中，总共有多少个batch
        num_batches_per_epoch = len(dataloader)

        for i, (images, roof, around_1, around_2) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            # if step < start_iter:
            #     # 如果有resume的需求，可以跳过已经训练过的step
            #     continue

            # 对文本数据进行tokenize并放到GPU
            roof = longclip.tokenize(roof, truncate=True).cuda()
            around_1 = longclip.tokenize(around_1, truncate=True).cuda()
            around_2 = longclip.tokenize(around_2, truncate=True).cuda()
            # short_text = longclip.tokenize(short_text, truncate=True).cuda()

            # 设置学习率（余弦退火+warmup）
            self.scheduler(step)

            # 每次iter前都要将梯度置0
            self.optimizer.zero_grad()

            # 自动混合精度环境
            with torch.cuda.amp.autocast():
                # 此处self.model(...)里封装了图像与文本的正向计算以及相似度计算
                # loss_long和loss_short是两个不同部分的损失
                loss_itr , loss_ita1 , loss_ita2 = self.model(images, roof, around_1, around_2, self.rank)
                loss = (loss_itr + loss_ita1 + loss_ita2)/3
                # loss = loss_long + loss_short  # 最终要回传的损失

            # 使用梯度缩放来避免混合精度下的溢出
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train(self, resume=False, warmup_length=200):
        """
        主训练函数，会调用train_epoch进行训练，并在每个epoch结束后进行测试。
        同时只保留测试结果最好的权重。
        """
        # 准备训练集和分布式采样器
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=32,
            pin_memory=True
        )

        # 构造余弦退火的学习率调度器
        self.scheduler = cosine_lr(
            self.optimizer,
            base_lr=self.lr,
            warmup_length=warmup_length,
            steps=self.num_epoch * len(train_loader)
        )

        start_epoch = 0
        resume_iter = 0

        for epoch in range(start_epoch, self.num_epoch):
            # 训练一个epoch
            self.train_epoch(train_loader, epoch, start_iter=resume_iter)

            # 每训练完一个epoch后，做一次测试
            recall_1_t2i, _, _, recall_1_i2t, _, _ = self.test(epoch=epoch)

            acc = (recall_1_i2t + recall_1_t2i)/2

            # 只有rank=0才需要做模型保存的操作
            if self.rank == 0:
                # 如果当前测试集准确率好于之前最佳，则更新并保存最好权重
                if acc > self.best_acc:
                    self.best_acc = acc
                    now = datetime.now()
                    formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                    best_ckpt_path = os.path.join(
                        self.ckptdir,
                        f"{formatted_date}_best_clip.pt"
                    )
                    torch.save(
                        self.model.module.state_dict(),
                        best_ckpt_path
                    )
                    print("=====================================")
                    print(f"New best recall_1 = {acc}, weights saved to {best_ckpt_path}")
                    print("=====================================")

def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    Support both slurm and torch.distributed.launch.
    Falls back to single GPU mode if no distributed environment is detected.
    """
    num_gpus = torch.cuda.device_count()

    # Detect if running in a SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # Set environment variables for master address and port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)

    # Detect if running with torch.distributed.launch
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback to single GPU mode
        print("No distributed environment detected, defaulting to single GPU mode.")
        rank = 0
        world_size = 1
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    # Set the current GPU device
    torch.cuda.set_device(rank % num_gpus)

    # Initialize process group for distributed training
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )

    return rank, rank % num_gpus


if __name__ == "__main__":
    """
    入口函数，解析命令行参数并启动训练。
    """
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='wd.')
    # parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument('--log_scale', default=2.3026, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=10, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per gpu.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--resume", default=False, action='store_true', help="resume training from checkpoint.")
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")

    args = parser.parse_args()

    # 分布式环境初始化
    rank, local_rank = setup_distributed()
    print("DDP Done")

    # 构造并启动训练器
    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank,
        args=args
    )

    # 开始训练
    trainer.train(
        resume=args.resume,
        warmup_length=args.warmup_length
    )
