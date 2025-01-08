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

def create_empty_desc(device):
    return {
        'input_ids': torch.tensor([], device=device),
        'attention_mask': torch.tensor([], device=device),
        'token_type_ids': torch.tensor([], device=device),
    }
