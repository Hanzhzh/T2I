import torch
from torch.amp import autocast, GradScaler
import utils  # 确保 utils 包含 MetricLogger 和 SmoothedValue
from loss import NCELoss  # 假设 NCELoss 位于 nce_loss 模块中

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
