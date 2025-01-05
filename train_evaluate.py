import torch
import torch.nn.functional as F
import gc
import time
import datetime
from collections import defaultdict
from tqdm import tqdm

from torch.amp import autocast, GradScaler
import utils  # 确保 utils 包含 MetricLogger 和 SmoothedValue
from loss import NCELoss  # 假设 NCELoss 位于 nce_loss 模块中
from loss import infoNCE

def train_v1(model, train_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    """
    优化后的训练函数
    """
    model.train()  # 设置模型为训练模式
    # torch.autograd.set_detect_anomaly(True)  # 仅在调试时启用

    # 初始化度量记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_2', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_3', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train Epoch: [{epoch}]'
    print_freq = 50  # 打印频率

    # 初始化加权 InfoNCE 损失函数，并将其移动到设备
    weighted_NCELoss = NCELoss.WeightedInfoNCELoss().to(device)

    # 使用混合精度训练
    scaler = GradScaler()

    processed_batches = 0  # 记录成功处理的批次数

    for batch_id, (image, desc_text,subset_id) in enumerate(
            metric_logger.log_every(train_loader, print_freq, header)):

        image = image.to(device)  # 将图像数据转移到设备

        # 对文本进行分词，保持原始字典格式
        desc_tokens = []
        for desc_text_dict in desc_text:
            desc_token = {}
            for key, descriptions in desc_text_dict[0].items():
                desc_token[key] = {}
                combined_caption = ""  # 初始化拼接的描述内容
                for desc_key, text in descriptions.items():
                    # 对单个描述进行分词
                    tokenized_caption = tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,  # 设置截断
                        max_length=config['max_tokens'],
                        return_tensors="pt"
                    )
                    # 存储结果
                    desc_token[key][desc_key] = tokenized_caption.to(device)
                    # 拼接描述
                    combined_caption += text + " "  # 以空格分隔不同描述
                # 将拼接后的描述加入字典
                combined_tokenized_caption = tokenizer(
                    combined_caption.strip(),
                    padding='max_length',
                    truncation=True,  # 设置截断
                    max_length=config['max_tokens'],
                    return_tensors="pt"
                )
                desc_token[key]["caption"] = combined_tokenized_caption.to(device)

            desc_tokens.append([desc_token])
            print("-----------------------")
            print(f"{desc_tokens[0]}")
            print("-----------------------")

        # 混合精度上下文
        with autocast(device_type=device.type):
            # 模型前向传播
            image_embeds, desc_embeds = model(image, desc_tokens, device)
            key_list = ["desc_roof", "desc_around", "caption"]

            # 确保每个文本嵌入是 2D 张量
            for key in desc_embeds:
                if desc_embeds[key].dim() == 3:
                    desc_embeds[key] = desc_embeds[key].squeeze(1)  # [batch_size, feature_dim]
                elif desc_embeds[key].dim() == 2:
                    pass  # 已经是二维张量，不需要处理
                else:
                    raise ValueError(f"Unexpected dimension {desc_embeds[key].dim()} for desc_embeds[{key}]")

            # 计算损失
            # batch_size_train = config['batch_size_train']
            try:
                # loss_1 = weighted_NCELoss(image_embeds, desc_embeds[key_list[0]][:batch_size_train], subset_id)
                # loss_2 = weighted_NCELoss(image_embeds, desc_embeds[key_list[1]][:batch_size_train], subset_id)
                # loss_3 = weighted_NCELoss(image_embeds, desc_embeds[key_list[2]][:batch_size_train], subset_id)
                loss_1 = weighted_NCELoss(image_embeds, desc_embeds[key_list[0]], subset_id)
                loss_2 = weighted_NCELoss(image_embeds, desc_embeds[key_list[1]], subset_id)
                loss_3 = weighted_NCELoss(image_embeds, desc_embeds[key_list[2]], subset_id)
                loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_3
            except KeyError as e:
                print(f"KeyError: {e}. 请检查 key_list 和 desc_embeds 的键是否匹配。")
                continue
            except RuntimeError as e:
                print(f"RuntimeError: {e}. 请检查张量的形状是否正确。")
                continue

            # 检查损失是否计算成功
            if loss_1 is None or loss_2 is None or loss_3 is None:
                print("某些损失计算结果为 None，跳过此批次。")
                continue

        # 反向传播和优化步骤
        optimizer.zero_grad()  # 清空梯度
        scaler.scale(loss).backward()  # 反向传播，缩放损失
        scaler.step(optimizer)  # 更新优化器
        scaler.update()  # 更新 scaler
        scheduler.step()  # 更新学习率调度器

        # 更新度量
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_1=loss_1.item())
        metric_logger.update(loss_2=loss_2.item())
        metric_logger.update(loss_3=loss_3.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        processed_batches += 1  # 成功处理的批次数增加

    # 打印平均度量
    print("Averaged stats:", metric_logger.global_avg())

    # 返回平均的度量
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def create_empty_desc(device):
    return {
        'input_ids': torch.tensor([], device=device),
        'attention_mask': torch.tensor([], device=device),
        'token_type_ids': torch.tensor([], device=device),
    }

def train_v2(model, train_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    """
    优化后的训练函数

    参数:
    - model: 需要训练的模型
    - train_loader: 训练数据加载器
    - optimizer: 优化器
    - tokenizer: 分词器，用于处理文本数据
    - epoch: 当前训练的轮次
    - device: 设备（CPU 或 GPU）
    - scheduler: 学习率调度器
    - config: 配置字典，包含如 'max_tokens' 等参数
    - args: 其他命令行参数，如梯度累积步数
    """
    model.train()  # 设置模型为训练模式
    torch.backends.cudnn.benchmark = True  # 如果输入大小固定，启用 cudnn 基准测试以提高性能

    # 初始化度量记录器，用于记录训练过程中的各种指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 学习率
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))  # 总损失
    metric_logger.add_meter('loss_1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))  # 子损失1
    metric_logger.add_meter('loss_2', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))  # 子损失2
    metric_logger.add_meter('loss_3', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))  # 子损失3

    header = f'Train Epoch: [{epoch}]'  # 训练轮次头部信息
    print_freq = 50  # 日志打印频率
    # 初始化加权 InfoNCE 损失函数，并将其移动到指定设备
    weighted_NCELoss = NCELoss.WeightedInfoNCELoss().to(device)
    # 初始化混合精度训练的梯度缩放器
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
    # 获取梯度累积步数，默认设置为1（不累积）
    accumulation_steps = config['accumulation_steps']

    # 遍历训练数据加载器
    for batch_id, (image, desc_text, subset_id) in enumerate(
            metric_logger.log_every(train_loader, print_freq, header)):

        # 将图像数据转移到指定设备，并启用非阻塞传输以提高效率
        image = image.to(device, non_blocking=True)
        # 初始化描述令牌的列表，每个元素对应一个样本
        desc_tokens = []
        image_names = []
        # 初始化各描述键的文本列表
        desc_roof_texts = []
        desc_around_texts = []
        captions = []

        # 通过索引来获取每个样本的 subset_id，非常重要！！！！
        for i, desc_text_dict in enumerate(desc_text):
            # current_subset_id = subset_id[i]
            first_dict = desc_text_dict[0]
            for image_filename, descriptions in first_dict.items():
                desc_roof = descriptions.get('desc_roof')
                desc_around = descriptions.get('desc_around')
                # 简单拼接两个字符串，中间添加句点和空格
                caption = desc_roof + ". " + desc_around if desc_roof and desc_around else desc_roof or desc_around
                caption = caption.strip()

                image_names.append(image_filename)
                desc_roof_texts.append(desc_roof)
                desc_around_texts.append(desc_around)
                captions.append(caption)
                desc_tokens.append([
                    {
                        image_filename: {
                            'desc_roof': create_empty_desc(device),
                            'desc_around': create_empty_desc(device),
                            'caption': create_empty_desc(device),
                        }
                    }
                ])

        assert len(image_names) == len(
            desc_roof_texts), f"长度不一致: image_names({len(image_names)}) != desc_roof_texts({len(desc_roof_texts)})"
        assert len(image_names) == len(
            desc_around_texts), f"长度不一致: image_names({len(image_names)}) != desc_around_texts({len(desc_around_texts)})"
        assert len(image_names) == len(
            captions), f"长度不一致: image_names({len(image_names)}) != captions({len(captions)})"
        assert len(image_names) == len(
            subset_id), f"长度不一致: image_names({len(image_names)}) != subset_id({len(subset_id)})"

        # 批量分词 'desc_roof' 描述
        tokenized_desc_roof = tokenizer(
            desc_roof_texts,
            padding='max_length',
            truncation=True,
            max_length=config['max_tokens'],
            return_tensors="pt"
        ).to(device)

        # 批量分词 'desc_around' 描述
        tokenized_desc_around = tokenizer(
            desc_around_texts,
            padding='max_length',
            truncation=True,
            max_length=config['max_tokens'],
            return_tensors="pt"
        ).to(device)

        # 批量分词 'caption' 描述
        tokenized_captions = tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=config['max_tokens'],
            return_tensors="pt"
        ).to(device)

        # print(f"\nSample {1}:")
        # print(f"  input_ids: {tokenized_captions['input_ids'][0]}")
        # print(f"  attention_mask: {tokenized_captions['attention_mask'][0]}")
        # if 'token_type_ids' in tokenized_captions:
        #     print(f"  token_type_ids: {tokenized_captions['token_type_ids'][0]}")
        # decoded_text = tokenizer.decode(tokenized_captions['input_ids'][0], skip_special_tokens=True)
        # print(f"  decoded_text: {decoded_text}")

        assert len(tokenized_desc_roof) == len(tokenized_captions) == len(tokenized_desc_around), "对文本进行tokenized的长度不一致，检查"

        # 将分词结果分配回 desc_tokens
        for i in range(len(desc_text)):
            if desc_roof_texts[i] == "" and desc_around_texts[i] == "":
                print("出现错误，文本为空")
                break

            # print(f"Assigning tokenized_desc_roof for sample {i}, image: {image_names[i]}")
            # print(f"image_names:{image_names}")

            # 检查 'token_type_ids' 是否存在
            if 'token_type_ids' in tokenized_desc_roof:
                token_type_ids = tokenized_desc_roof['token_type_ids'][i]
            else:
                print("提醒: 'token_type_ids' 不存在于 tokenized_desc_roof 中，使用全零张量作为默认值。")
                token_type_ids = torch.zeros_like(tokenized_desc_roof['input_ids'][i])

            # 赋值给 desc_tokens
            desc_tokens[i][0][image_names[i]]['desc_roof'] = {
                'input_ids': tokenized_desc_roof['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_desc_roof['attention_mask'][i].unsqueeze(0),
                'token_type_ids': token_type_ids.unsqueeze(0)
            }

            desc_tokens[i][0][image_names[i]]['desc_around'] = {
                'input_ids': tokenized_desc_around['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_desc_around['attention_mask'][i].unsqueeze(0),
                'token_type_ids': tokenized_desc_around['token_type_ids'][
                    i].unsqueeze(0) if 'token_type_ids' in tokenized_desc_around else torch.zeros_like(
                    tokenized_desc_around['input_ids'][i].unsqueeze(0))
            }
            desc_tokens[i][0][image_names[i]]['caption'] = {
                'input_ids': tokenized_captions['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_captions['attention_mask'][i].unsqueeze(0),
                'token_type_ids': tokenized_captions['token_type_ids'][
                    i].unsqueeze(0) if 'token_type_ids' in tokenized_captions else torch.zeros_like(
                    tokenized_captions['input_ids'][i].unsqueeze(0))
            }

        # 启用自动混合精度上下文管理器
        with autocast(device_type=device.type):
            # 模型前向传播，获取图像和描述的嵌入向量
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(f"image_shape:{image.shape}")
            # print(f"desc_tokens shape:{len(desc_tokens)}")
            # print(f"desc_tokens[0]:{desc_tokens[0]}")
            image_embeds, desc_embeds = model(image, desc_tokens)

            key_list = ["desc_roof", "desc_around", "caption"]  # 需要计算损失的描述键列表

            # 确保每个文本嵌入是二维张量
            for key in desc_embeds:
                if desc_embeds[key].dim() == 3:
                    desc_embeds[key] = desc_embeds[key].squeeze(1)  # 压缩多余的维度
                elif desc_embeds[key].dim() == 2:
                    pass  # 已经是二维张量，无需处理
                else:
                    raise ValueError(f"Unexpected dimension {desc_embeds[key].dim()} for desc_embeds[{key}]")

            # print("************************************************************************")
            # print(f"image_embeds.shape:{image_embeds.shape}")
            # print(f"desc_embeds[key_list[0]]:{desc_embeds[key_list[0]]}")
            # print(f"image_names:{image_names}")
            # print("************************************************************************")
            try:
                # 计算各个描述的损失
                loss_1 = weighted_NCELoss(image_embeds, desc_embeds[key_list[0]], image_names)
                loss_2 = weighted_NCELoss(image_embeds, desc_embeds[key_list[1]], image_names)
                loss_3 = weighted_NCELoss(image_embeds, desc_embeds[key_list[2]], image_names)
                # 加权总损失
                loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_3
            except KeyError as e:
                print(f"KeyError: {e}. 请检查 key_list 和 desc_embeds 的键是否匹配。")
                break  # 跳过当前批次，继续下一个
            except RuntimeError as e:
                print(f"RuntimeError: {e}. 请检查张量的形状是否正确。")
                break  # 跳过当前批次，继续下一个

            # 检查损失是否成功计算
            if loss_1 is None or loss_2 is None or loss_3 is None:
                print("某些损失计算结果为 None，跳过此批次。")
                break  # 跳过当前批次，继续下一个

        # 反向传播和优化步骤
        scaler.scale(loss).backward()  # 反向传播，缩放损失以防止梯度下溢
        # 在 train_v2 中详细检查梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"参数 {name} 的梯度范数: {grad_norm}")
        #         if grad_norm == 0:
        #             print(f"警告: 参数 {name} 的梯度为零")
        #     else:
        #         print(f"警告: 参数 {name} 的梯度为 None")


        # 梯度累积逻辑：只有在达到累积步数时才更新模型参数
        if (batch_id + 1) % accumulation_steps == 0:
            scaler.step(optimizer)      # 更新优化器参数
            scaler.update()             # 更新缩放器
            optimizer.zero_grad()       # 清空梯度
            scheduler.step()            # 更新学习率调度器

        # 更新度量记录器
        metric_logger.update(loss=loss.item())      # 更新总损失
        metric_logger.update(loss_1=loss_1.item())  # 更新子损失1
        metric_logger.update(loss_2=loss_2.item())  # 更新子损失2
        metric_logger.update(loss_3=loss_3.item())  # 更新子损失3
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # 更新学习率

        # 及时释放未使用的 CUDA 缓存，以减少显存碎片
        torch.cuda.empty_cache()

    # 打印平均度量统计
    print("Averaged stats:", metric_logger.global_avg())

    # 返回平均的度量结果，格式化为字符串
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

from torch.nn.utils import clip_grad_norm_

def train_v3(model, train_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    """
    简化后的训练函数，使用标准 InfoNCE 损失进行验证。
    """
    model.train()
    torch.backends.cudnn.benchmark = True

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train Epoch: [{epoch}]'
    print_freq = 50

    # 使用标准 InfoNCE 损失
    info_nce_loss = infoNCE.InfoNCE_Loss(temperature=config.get('temperature', 0.07)).to(device)

    scaler = GradScaler(enabled=config.get('use_amp', False))
    accumulation_steps = config.get('accumulation_steps', 1)
    optimizer.zero_grad()

    for batch_id, (image, desc_text, subset_id) in enumerate(
            metric_logger.log_every(train_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)

        # 初始化描述令牌的列表，每个元素对应一个样本
        desc_tokens = []
        image_names = []
        desc_roof_texts = []
        desc_around_texts = []
        captions = []

        for i, desc_text_dict in enumerate(desc_text):
            first_dict = desc_text_dict[0]
            for image_filename, descriptions in first_dict.items():
                desc_roof = descriptions.get('desc_roof')
                desc_around = descriptions.get('desc_around')
                caption = desc_roof + ". " + desc_around if desc_roof and desc_around else desc_roof or desc_around
                caption = caption.strip()

                image_names.append(image_filename)
                desc_roof_texts.append(desc_roof)
                desc_around_texts.append(desc_around)
                captions.append(caption)
                desc_tokens.append([
                    {
                        image_filename: {
                            'desc_roof': create_empty_desc(device),
                            'desc_around': create_empty_desc(device),
                            'caption': create_empty_desc(device),
                        }
                    }
                ])

        # 批量分词 'desc_roof', 'desc_around', 'caption' 描述
        tokenized_desc_roof = tokenizer(
            desc_roof_texts,
            padding='max_length',
            truncation=True,
            max_length=config.get('max_tokens', 128),
            return_tensors="pt"
        ).to(device)

        tokenized_desc_around = tokenizer(
            desc_around_texts,
            padding='max_length',
            truncation=True,
            max_length=config.get('max_tokens', 128),
            return_tensors="pt"
        ).to(device)

        tokenized_captions = tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=config.get('max_tokens', 128),
            return_tensors="pt"
        ).to(device)

        # 检查分词结果的长度一致性
        assert len(tokenized_desc_roof['input_ids']) == len(tokenized_captions['input_ids']) == len(tokenized_desc_around['input_ids']), "分词长度不一致"

        # 将分词结果分配回 desc_tokens
        for i in range(len(desc_text)):
            if desc_roof_texts[i] == "" and desc_around_texts[i] == "":
                print("出现错误，文本为空")
                break

            token_type_ids_roof = tokenized_desc_roof['token_type_ids'][i] if 'token_type_ids' in tokenized_desc_roof else torch.zeros_like(tokenized_desc_roof['input_ids'][i])
            token_type_ids_around = tokenized_desc_around['token_type_ids'][i] if 'token_type_ids' in tokenized_desc_around else torch.zeros_like(tokenized_desc_around['input_ids'][i])
            token_type_ids_caption = tokenized_captions['token_type_ids'][i] if 'token_type_ids' in tokenized_captions else torch.zeros_like(tokenized_captions['input_ids'][i])

            desc_tokens[i][0][image_names[i]]['desc_roof'] = {
                'input_ids': tokenized_desc_roof['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_desc_roof['attention_mask'][i].unsqueeze(0),
                'token_type_ids': token_type_ids_roof.unsqueeze(0)
            }

            desc_tokens[i][0][image_names[i]]['desc_around'] = {
                'input_ids': tokenized_desc_around['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_desc_around['attention_mask'][i].unsqueeze(0),
                'token_type_ids': token_type_ids_around.unsqueeze(0)
            }

            desc_tokens[i][0][image_names[i]]['caption'] = {
                'input_ids': tokenized_captions['input_ids'][i].unsqueeze(0),
                'attention_mask': tokenized_captions['attention_mask'][i].unsqueeze(0),
                'token_type_ids': token_type_ids_caption.unsqueeze(0)
            }

        with autocast(enabled=config.get('use_amp', False), device_type='cuda'):
            # 模型前向传播，获取图像和描述的嵌入向量
            image_embeds, desc_embeds = model(image, desc_tokens)
            text_embeds = desc_embeds["caption"]

            # 检查嵌入是否包含 NaN 或 Inf
            if torch.isnan(image_embeds).any() or torch.isinf(image_embeds).any():
                print("警告：image_embeds 包含 NaN 或 Inf")
            if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
                print("警告：text_embeds 包含 NaN 或 Inf")

                # 打印嵌入的统计信息
            print(
                f"image_embeds: min={image_embeds.min().item()}, max={image_embeds.max().item()}, mean={image_embeds.mean().item()}")
            print(
                f"text_embeds: min={text_embeds.min().item()}, max={text_embeds.max().item()}, mean={text_embeds.mean().item()}")

            # 计算损失
            loss = info_nce_loss(image_embeds, text_embeds)

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                print("警告：损失值为 NaN")

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积逻辑：只有在达到累积步数时才取消缩放、裁剪梯度和更新参数
        if (batch_id + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 检查梯度是否为 NaN
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"警告：{name} 的梯度包含 NaN")
                    print(f"{name} 梯度范数: {param.grad.norm().item()}")
                    break  # 仅检查第一个有梯度的参数

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # 打印参数变化（可选）
            for name, param in model.named_parameters():
                if name == "vision_encoder.patch_embed.proj.weight":
                    if param.data.dim() > 0:
                        print(f"更新后 {name} 的第一个元素: {param.data.view(-1)[0].item()}")
                    else:
                        print(f"更新后 {name} 的值: {param.data.item()}")
                    break

        # 更新度量记录器
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 释放未使用的 CUDA 缓存
        torch.cuda.empty_cache()

    # 打印平均度量统计
    print("Averaged stats:", metric_logger.global_avg())

    # 返回平均的度量结果，格式化为字符串
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation_or(model, test_loader, tokenizer, device, config):
    """
    评估函数，计算每张图像与 desc_text 中每个描述的双向得分（i2t 和 t2i）。
    """
    model = model.half()  # 转为半精度
    model = model.eval()  # 设置为评估模式

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    # 准备列表，用于收集所有图像的embedding，以及路径、索引
    all_image_embeds = []
    all_image_paths = []

    # 准备一个字典，用于存放多种描述类别的文本embedding
    desc_dict = {}

    for batch_id, (image, desc_text, image_path, json_path, subset_id, sample_index) in enumerate(
            metric_logger.log_every(test_loader, 2, header)):

        image = image.to(torch.float16).to(device)

        desc_tokens = []
        for i in range(len(desc_text)):
            desc_text_dict = desc_text[i]
            desc_token = {}
            for key, descriptions in desc_text_dict[0].items():
                desc_token[key] = {}
                combined_caption = ""  # 初始化拼接的描述内容
                for desc_key, text in descriptions.items():
                    # 对单个描述进行分词
                    tokenized_caption = tokenizer(
                        text,
                        max_length=config['max_tokens'],
                        padding='max_length',
                        truncation=True,  # 设置截断
                        return_tensors="pt"
                    ).to(device)
                    # 存储结果
                    desc_token[key][desc_key] = tokenized_caption
                    # 拼接描述
                    combined_caption += text + " "  # 以空格分隔不同描述
                # 将拼接后的描述加入字典
                combined_tokenized_caption = tokenizer(
                    combined_caption.strip(),
                    max_length=config['max_tokens'],
                    padding='max_length',
                    truncation=True,  # 设置截断
                    return_tensors="pt"
                ).to(device)
                desc_token[key]["caption"] = combined_tokenized_caption

            desc_tokens.append([desc_token])  # desc_tokens包含id att

        # 模型前向传播
        image_embeds, desc_embeds, subset_id, key_list = model(image, desc_tokens, image_path, json_path, subset_id,
                                                               sample_index)

        # 确保每个文本嵌入是 2D 张量
        for key in desc_embeds:
            if desc_embeds[key].dim() == 3:
                desc_embeds[key] = desc_embeds[key].squeeze(1)  # [batch_size, feature_dim]
                print(f"After squeezing, '{key}' 的维度: {desc_embeds[key].shape}")
            elif desc_embeds[key].dim() == 2:
                # 已经是二维张量，不需要处理
                pass
            else:
                raise ValueError(f"Unexpected dimension {desc_embeds[key].dim()} for desc_embeds[{key}]")

        # 将本批次的图像embedding保存
        all_image_embeds.append(image_embeds)
        all_image_paths.extend(image_path)

        # 将 desc_embeds 中的多类描述embedding 根据 key_list 依次拼接
        for desc_key in key_list:
            key_name = f"desc_embeds_{desc_key}"
            if key_name not in desc_dict:
                desc_dict[key_name] = []
            desc_dict[key_name].append(desc_embeds[key_name])

    # 将所有batch拼接起来
    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N_img, embed_dim]
    for k in desc_dict.keys():
        desc_dict[k] = torch.cat(desc_dict[k], dim=0)  # [N_img, embed_dim]

    # 打印一下大小
    print(f"Collected {all_image_embeds.shape[0]} image embeddings.")
    for k, v in desc_dict.items():
        print(f"Collected {v.shape[0]} embeddings for {k}.")

    print("Image embedding over ...")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n评估耗时 {total_time_str}')

    return desc_dict  # 根据需要返回适当的内容


@torch.no_grad()
def evaluation_simple_amp(model, data_loader, tokenizer, device, config):
    """
    Simplified evaluation function using Automatic Mixed Precision (AMP): Compute similarity between image and text features and calculate recall@K.
    """
    model.eval()
    model.to(device)

    all_image_embeds = []
    text_embeds_dict = defaultdict(list)

    print('开始计算评估特征...')
    start_time = time.time()

    # 创建一个自动混合精度的上下文管理器
    scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_amp', False))

    for batch in tqdm(data_loader, desc="Processing Batches"):
        images, desc_text, _, _, _, _ = batch

        images = images.to(device, dtype=torch.float32, non_blocking=True)  # 保持输入为float32

        with torch.cuda.amp.autocast(enabled=config.get('use_amp', False)):
            image_feat = model.vision_encoder(images)  # [bs, num_patches, vision_width]
            image_embed = model.vision_proj(image_feat[:, 0, :])  # [bs, embed_dim]
            image_embed = F.normalize(image_embed, dim=-1)

        all_image_embeds.append(image_embed.cpu())

        for desc in desc_text:
            desc_dict = desc[0]
            for _, desc_items in desc_dict.items():
                for desc_key, desc_str in desc_items.items():
                    tokenized = tokenizer(
                        desc_str,
                        padding='max_length',
                        truncation=True,
                        max_length=config['max_tokens'],
                        return_tensors="pt"
                    ).to(device)

                    with torch.cuda.amp.autocast(enabled=config.get('use_amp', False)):
                        text_embed = model.text_proj(model.text_encoder.embeddings(tokenized.input_ids))[:, 0, :]
                        text_embed = F.normalize(text_embed, dim=-1)

                    text_embeds_dict[desc_key].append(text_embed.cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0).to(device)  # [num_images, embed_dim]
    print(f"Total image embeddings shape: {image_embeds.shape}")

    del all_image_embeds
    gc.collect()

    recall_dict = {}

    for desc_key, text_embeds_list in text_embeds_dict.items():
        print(f"\n=== Processing Description Category: {desc_key} ===")

        text_embeds = torch.cat(text_embeds_list, dim=0).to(device)  # [num_texts, embed_dim]

        sims_matrix = image_embeds @ text_embeds.t()  # [num_images, num_texts]
        print(f"Similarity matrix shape for {desc_key}: {sims_matrix.shape}")

        recall_at_k = {}
        for k in config['recall_k']:
            # 图像到文本 (i2t)
            i2t_topk = sims_matrix.topk(k, dim=1).indices  # [num_images, k]
            correct_i2t = torch.arange(sims_matrix.size(0)).unsqueeze(1).to(device)  # [num_images, 1]
            hits_i2t = (i2t_topk == correct_i2t).any(dim=1).float()  # [num_images]
            recall_i2t = hits_i2t.mean().item()

            # 文本到图像 (t2i)
            t2i_topk = sims_matrix.topk(k, dim=0).indices  # [k, num_texts]
            correct_t2i = torch.arange(sims_matrix.size(1)).unsqueeze(0).to(device)  # [1, num_texts]
            hits_t2i = (t2i_topk == correct_t2i).any(dim=0).float()  # [num_texts]
            recall_t2i = hits_t2i.mean().item()

            recall_at_k[k] = {
                'i2t': recall_i2t,
                't2i': recall_t2i
            }
            print(f"Recall@{k} - i2t: {recall_i2t:.4f}, t2i: {recall_t2i:.4f}")

        recall_dict[desc_key] = recall_at_k

        del text_embeds, sims_matrix, i2t_topk, t2i_topk
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n评估耗时 {total_time_str}')

    return recall_dict


@torch.no_grad()
def evaluation_simple_amp_optimized(model, data_loader, tokenizer, device, config):
    """
    优化后的评估函数:
    1) 分批次计算图像、文本特征并存储到 CPU;
    2) 分块计算相似度和 Recall@K, 避免一次性创建过大的相似度矩阵.
    """
    model.eval()
    model.to(device)

    # ------------------------------------------------------------
    # 1) 准备阶段：定义一些列表/字典，用于收集结果
    # ------------------------------------------------------------
    all_image_embeds = []
    text_embeds_dict = defaultdict(list)

    print('开始计算评估特征...')
    start_time = time.time()

    # 是否使用自动混合精度
    use_amp = config.get('use_amp', False)

    # ------------------------------------------------------------
    # 2) 逐批次计算图像特征并收集到 CPU
    # ------------------------------------------------------------
    # 在这里只做图像特征提取, 不在同一循环里处理文本, 可以减少 GPU 占用
    for batch in tqdm(data_loader, desc="提取图像特征："):
        images, desc_text, _ = batch
        images = images.to(device, dtype=torch.float32, non_blocking=True)  # 输入保持 float32

        with torch.cuda.amp.autocast(enabled=use_amp):
            image_feat = model.vision_encoder(images)  # [bs, num_patches, vision_width]
            image_embed = model.vision_proj(image_feat[:, 0, :])  # [bs, embed_dim]
            image_embed = F.normalize(image_embed, dim=-1)

        # 将图像嵌入从 GPU 转移到 CPU, 并追加到 all_image_embeds
        all_image_embeds.append(image_embed.cpu())

        # 释放 GPU 占用
        del images, desc_text, image_feat, image_embed
        gc.collect()
        torch.cuda.empty_cache()

    # 在这里合并所有图像嵌入到一个大 tensor (仍在 CPU 上)
    image_embeds = torch.cat(all_image_embeds, dim=0)  # [num_images, embed_dim]
    del all_image_embeds
    gc.collect()

    print(f"总的图像嵌入的shape: {image_embeds.shape}")

    # ------------------------------------------------------------
    # 3) 重新遍历数据集，仅提取文本特征(也可以单独构建一个文本特征的 data_loader)
    # ------------------------------------------------------------
    # 如果文本数据量同样庞大，也可以拆分或使用单独的循环来收集
    for batch in tqdm(data_loader, desc="提取文本特征："):
        _, desc_text, _ = batch

        # 对于本批次内的所有描述，统一处理
        for desc in desc_text:
            desc_dict = desc[0]
            for _, desc_items in desc_dict.items():
                for desc_key, desc_str in desc_items.items():
                    tokenized = tokenizer(
                        desc_str,
                        padding='max_length',
                        truncation=True,
                        max_length=config['max_tokens'],
                        return_tensors="pt"
                    ).to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        text_embed = model.text_proj(
                            model.text_encoder.embeddings(tokenized.input_ids)
                        )[:, 0, :]
                        text_embed = F.normalize(text_embed, dim=-1)

                    # 将文本嵌入转移到 CPU
                    text_embeds_dict[desc_key].append(text_embed.cpu())

        # 释放 GPU 占用
        del desc_text
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 4) 计算 Recall@K，需要用到图像-文本相似度矩阵
    #    但为了减少 GPU 占用, 我们可以分块计算相似度
    # ------------------------------------------------------------
    # 先将 image_embeds 放回 GPU (如果 GPU 内存足够的话)
    image_embeds = image_embeds.to(device)
    recall_dict = {}

    # 这里设定一个分块大小, 避免一次性占用过大显存
    # 数值需要根据实际 GPU 情况以及数据维度来调节
    chunk_size = config["chunk_size"]

    for desc_key, text_embeds_list in text_embeds_dict.items():
        print(f"\n=== 处理描述类别: {desc_key} ===")

        # 将所有文本特征合并 (先在 CPU 上)
        text_embeds = torch.cat(text_embeds_list, dim=0)  # [num_texts, embed_dim]
        # 再放入 GPU
        text_embeds = text_embeds.to(device)

        num_images = image_embeds.shape[0]
        num_texts = text_embeds.shape[0]

        # 相似度矩阵大小: [num_images, num_texts]
        # 我们分块(针对图像维度或文本维度)进行乘法, 避免一次性创建过大矩阵
        sims_matrix = torch.zeros((num_images, num_texts), dtype=torch.float32, device=device)

        # --------------------------------------------------------
        # 分块计算 (示例：以图像维度进行chunk)
        # --------------------------------------------------------
        for i_start in range(0, num_images, chunk_size):
            i_end = min(i_start + chunk_size, num_images)
            # 取出部分图像特征 [chunk_size, embed_dim]
            sub_image_embeds = image_embeds[i_start:i_end, :]  # GPU 上

            # 计算这部分图像与所有文本的相似度 [chunk_size, num_texts]
            sub_sims = sub_image_embeds @ text_embeds.t()
            # 写入 sims_matrix 对应区域
            sims_matrix[i_start:i_end, :] = sub_sims

        # 现在已经拿到了完整的相似度矩阵 sims_matrix: [num_images, num_texts]

        print(f"Similarity matrix shape for {desc_key}: {sims_matrix.shape}")

        # --------------------------------------------------------
        # 5) 计算 Recall@K
        # --------------------------------------------------------
        recall_at_k = {}
        for k in config['recall_k']:
            # 图像到文本 (i2t)
            i2t_topk = sims_matrix.topk(k, dim=1).indices  # [num_images, k]
            correct_i2t = torch.arange(num_images, device=device).unsqueeze(1)  # [num_images, 1]
            hits_i2t = (i2t_topk == correct_i2t).any(dim=1).float()
            recall_i2t = hits_i2t.mean().item()

            # 文本到图像 (t2i)
            t2i_topk = sims_matrix.topk(k, dim=0).indices  # [k, num_texts]
            correct_t2i = torch.arange(num_texts, device=device).unsqueeze(0)  # [1, num_texts]
            hits_t2i = (t2i_topk == correct_t2i).any(dim=0).float()
            recall_t2i = hits_t2i.mean().item()

            recall_at_k[k] = {
                'i2t': recall_i2t,
                't2i': recall_t2i
            }
            print(f"Recall@{k} - i2t: {recall_i2t:.4f}, t2i: {recall_t2i:.4f}")

        recall_dict[desc_key] = recall_at_k

        # 释放相关显存
        del text_embeds, sims_matrix, i2t_topk, t2i_topk
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n评估耗时 {total_time_str}')

    return recall_dict


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    """
    评估函数（区分不同描述类别版本）。

    - 每张图像可以含多个描述类别（如 "caption", "Description1", "Description2", ...）
    - 在评估时，把每个描述类别分别当作一个独立的文本池，
      分开生成相似度矩阵并进行跨模态融合打分，得到 i2t、t2i 的精细得分矩阵。

    返回：
    --------
    score_dict : dict，形如
      {
        'caption':     (score_matrix_i2t_caption, score_matrix_t2i_caption),
        'Description1':(score_matrix_i2t_descr1,  score_matrix_t2i_descr1),
        'Description2':...
        ...
      }
    用于后续计算 recall@K 等检索指标。
    """

    # 切换半精度 + eval模式
    model = model.half()
    model.eval()

    # 用于日志打印
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('开始计算评估特征...')
    start_time = time.time()

    # -------------------------------------------------------
    # 1) 收集所有图像特征
    # -------------------------------------------------------
    all_image_feats = []
    all_image_embeds = []
    image_size_total = 0

    # 2) 建立字典，用于存放不同描述类别的文本特征
    #    例如 text_feats_dict["caption"] = [...], text_feats_dict["Description1"] = [...], ...
    text_feats_dict = {}
    text_embeds_dict = {}
    text_atts_dict = {}

    # -------------------------------------------------------
    # 3) 遍历 data_loader，收集图像特征和分类别的文本特征
    # -------------------------------------------------------
    for batch_id, (images, desc_text, image_paths, json_paths, subset_id, sample_index) in enumerate(
            metric_logger.log_every(data_loader, 50, header)):

        # 3.1) 处理图像
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        # 调用模型视觉编码器
        image_feat = model.vision_encoder(images)  # [bs, num_patches, vision_width]
        image_embed = model.vision_proj(image_feat[:, 0, :])  # [bs, embed_dim]

        # 检查 image_embed 的形状
        print(f"Batch {batch_id}: image_embed shape before normalization: {image_embed.shape}")  # 应该是 [bs, embed_dim]

        # 确保 image_embed 是二维的
        if image_embed.dim() != 2:
            raise ValueError(f"Expected image_embed to be 2D, got {image_embed.dim()}D")

        image_embed = F.normalize(image_embed, dim=-1)  # [bs, embed_dim]

        # 再次检查 image_embed 的形状
        print(f"Batch {batch_id}: image_embed shape after normalization: {image_embed.shape}")  # 应该是 [bs, embed_dim]

        all_image_feats.append(image_feat)
        all_image_embeds.append(image_embed)

        image_size_total += images.size(0)

        # 3.2) 处理文本：desc_text 里包含多类描述
        # desc_text[i] ~ [{ "image-xxx.jpg": { "Description1": "...", "Description2": "...", "caption":"..." } }]
        # 假设 batch_size=bs => i in [0..bs-1]
        for i in range(len(desc_text)):
            desc_dict_for_one_image = desc_text[i][0]
            # desc_dict_for_one_image 可能是 { "image-xxx.jpg": { "Description1": "...", "Description2": "...", ... } }
            for _, desc_item in desc_dict_for_one_image.items():
                # desc_item: { "caption": "...", "Description1": "...", ...}
                for desc_key, desc_str in desc_item.items():
                    # 根据 desc_key 决定要放到哪个字典
                    if desc_key not in text_feats_dict:
                        text_feats_dict[desc_key] = []
                        text_embeds_dict[desc_key] = []
                        text_atts_dict[desc_key] = []

                    # 对该描述进行分词
                    tokenized = tokenizer(
                        desc_str,
                        padding='max_length',
                        truncation=True,
                        max_length=config['max_tokens'],
                        return_tensors="pt"
                    ).to(device)

                    # 获取输入嵌入而不是 last_hidden_state
                    input_embeddings = model.text_encoder.embeddings(tokenized.input_ids)  # [1, seq_len, embedding_dim]

                    # 检查 input_embeddings 的形状
                    print(f"Batch {batch_id}, Image {i}: input_embeddings shape: {input_embeddings.shape}")  # [1, seq_len, embedding_dim]

                    # 归一化并投影嵌入
                    text_embed = model.text_proj(input_embeddings[:, 0, :])  # [1, embed_dim]
                    text_embed = F.normalize(text_embed, dim=-1)  # [1, embed_dim]

                    # 检查 text_embed 的形状
                    print(f"Batch {batch_id}, Image {i}: text_embed shape: {text_embed.shape}")  # [1, embed_dim]

                    # 存储嵌入和注意力掩码
                    text_feats_dict[desc_key].append(input_embeddings.squeeze(0))  # [seq_len, embedding_dim]
                    text_embeds_dict[desc_key].append(text_embed.squeeze(0))       # [embed_dim]
                    text_atts_dict[desc_key].append(tokenized.attention_mask.squeeze(0))  # [seq_len]

    # 拼接图像特征
    image_embeds = torch.cat(all_image_embeds, dim=0)  # [num_images, embed_dim]
    image_feats = torch.cat(all_image_feats, dim=0)    # [num_images, num_patches, vision_width]
    print(f"image_embeds shape after concat: {image_embeds.shape}")  # 应该是 [40, 256]
    print(f"image_feats shape after concat: {image_feats.shape}")      # 应该是 [40, 512, 768]

    # 确保 image_embeds 是二维的，image_feats 是三维的
    if image_embeds.dim() != 2 or image_feats.dim() != 3:
        raise ValueError(f"Expected image_embeds to be 2D and image_feats to be 3D, got {image_embeds.dim()}D and {image_feats.dim()}D respectively.")

    num_images = image_embeds.size(0)

    # 保留 image_feats，删除 all_image_feats 和 all_image_embeds
    del all_image_embeds, all_image_feats
    gc.collect()
    torch.cuda.empty_cache()

    print(f"共收集了 {num_images} 张图像。")

    # -----------------------------------------------------------------
    # 4) 对每个描述类别分别进行：拼接所有文本特征 -> 计算相似度矩阵 -> 取TopK做跨模态融合 -> 得到score_matrix_i2t, score_matrix_t2i
    # -----------------------------------------------------------------
    score_dict = {}  # 最终返回的结果 { "caption": (score_matrix_i2t_cap, score_matrix_t2i_cap), ... }

    for desc_key in text_feats_dict.keys():
        print(f"\n=== 处理描述类别: {desc_key} ===")

        # 4.1) 拼接该描述类别的文本特征
        try:
            text_feats = torch.stack(text_feats_dict[desc_key], dim=0)  # [num_texts, seq_len, embedding_dim]
            text_embeds = torch.stack(text_embeds_dict[desc_key], dim=0)  # [num_texts, embed_dim]
        except Exception as e:
            print(f"Error stacking text_feats or text_embeds for {desc_key}: {e}")
            continue

        text_atts = torch.stack(text_atts_dict[desc_key], dim=0)      # [num_texts, seq_len]

        # 检查文本嵌入的形状
        print(f"text_embeds shape for {desc_key}: {text_embeds.shape}")  # 应该是 [num_texts, embed_dim]

        num_texts = text_feats.size(0)
        print(f"共收集了 {num_texts} 个文本描述，描述类别为 {desc_key}。")

        # 4.2) 计算初步相似度矩阵: [num_images, num_texts_total]
        try:
            sims_matrix = image_embeds @ text_embeds.t()
            print(f"sims_matrix shape for {desc_key}: {sims_matrix.shape}")  # 应该是 [num_images, num_texts_total]
        except Exception as e:
            print(f"Error computing sims_matrix for {desc_key}: {e}")
            continue

        # 构建评分矩阵
        score_matrix_i2t = torch.full((num_images, text_embeds.size(0)), -100.0, dtype=torch.float16, device=device)
        score_matrix_t2i = torch.full((text_embeds.size(0), num_images), -100.0, dtype=torch.float16, device=device)

        # 4.3) 图像到文本（i2t）评分
        header_local = f'评估 {desc_key}-i2t:'
        for i in range(num_images):
            sims = sims_matrix[i]
            # 对第 i 张图像取 TopK
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

            # 提取该图像的 [1, num_patches, vision_width]
            encoder_output = image_feats[i].unsqueeze(0).repeat(config['k_test'], 1, 1).to(device)  # [k_test, num_patches, vision_width]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            # 将TopK文本做cross-attention
            try:
                output = model.text_encoder(
                    encoder_embeds=text_feats[topk_idx].to(device),        # [k_test, seq_len, embedding_dim]
                    attention_mask=text_atts[topk_idx].to(device),        # [k_test, seq_len]
                    encoder_hidden_states=encoder_output,                # [k_test, num_patches, vision_width]
                    encoder_attention_mask=encoder_att,                  # [k_test, num_patches]
                    return_dict=True,
                    mode='fusion'
                )
            except Exception as e:
                print(f"Error during text_encoder fusion for i2t, desc_key={desc_key}, image_index={i}: {e}")
                continue

            # ITM头分数(正类的 logit)
            try:
                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]  # [k_test]
                score_matrix_i2t[i, topk_idx] = score.half()
            except Exception as e:
                print(f"Error during itm_head scoring for i2t, desc_key={desc_key}, image_index={i}: {e}")
                continue

        # 4.4) 文本到图像（t2i）评分
        sims_matrix_t = sims_matrix.t()
        del sims_matrix
        gc.collect()

        header_local = f'评估 {desc_key}-t2i:'
        for i in range(text_embeds.size(0)):
            sims = sims_matrix_t[i]
            # 取 TopK
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

            # [k_test, num_patches, vision_width]
            encoder_output = image_feats[topk_idx].to(device)  # [k_test, num_patches, vision_width]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            # 重复文本特征和注意力掩码
            repeated_text_feat = text_feats[i].unsqueeze(0).repeat(config['k_test'], 1, 1).to(device)  # [k_test, seq_len, embedding_dim]
            repeated_text_att = text_atts[i].unsqueeze(0).repeat(config['k_test'], 1).to(device)      # [k_test, seq_len]

            # 将TopK图像做cross-attention
            try:
                output = model.text_encoder(
                    encoder_embeds=repeated_text_feat,           # [k_test, seq_len, embedding_dim]
                    attention_mask=repeated_text_att,           # [k_test, seq_len]
                    encoder_hidden_states=encoder_output,       # [k_test, num_patches, vision_width]
                    encoder_attention_mask=encoder_att,         # [k_test, num_patches]
                    return_dict=True,
                    mode='fusion'
                )
            except Exception as e:
                print(f"Error during text_encoder fusion for t2i, desc_key={desc_key}, text_index={i}: {e}")
                continue

            # ITM头分数(正类的 logit)
            try:
                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]  # [k_test]
                score_matrix_t2i[i, topk_idx] = score.half()
            except Exception as e:
                print(f"Error during itm_head scoring for t2i, desc_key={desc_key}, text_index={i}: {e}")
                continue

        # 转回CPU，并保存在字典中
        try:
            score_i2t_np = score_matrix_i2t.cpu().numpy()
            score_t2i_np = score_matrix_t2i.cpu().numpy()
        except Exception as e:
            print(f"Error converting score matrices to numpy for {desc_key}: {e}")
            continue

        # 清理显存
        del score_matrix_i2t, score_matrix_t2i, sims_matrix_t
        gc.collect()
        torch.cuda.empty_cache()

        # 记录到总的score_dict
        score_dict[desc_key] = (score_i2t_np, score_t2i_np)

    # --------------------------------------------------------------------------------
    # 5) 返回每个描述类别对应的 i2t, t2i 的精细评分矩阵，用于后续指标计算
    # --------------------------------------------------------------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('评估耗时 {}'.format(total_time_str))

    return score_dict
