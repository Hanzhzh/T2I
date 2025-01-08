import torch
from torch.amp import autocast, GradScaler
import utils  # 确保 utils 包含 MetricLogger 和 SmoothedValue
from loss import NCELoss  # 假设 NCELoss 位于 nce_loss 模块中

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

        # # 打印文本分支梯度范数
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if "text_encoder" in name or "text_proj" in name:
        #             print(f"{name} 梯度范数: {param.grad.norm().item()}")
        #     else:
        #         if "text_encoder" in name or "text_proj" in name:
        #             print(f"{name} 没有梯度")
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



def create_empty_desc(device):
    return {
        'input_ids': torch.tensor([], device=device),
        'attention_mask': torch.tensor([], device=device),
        'token_type_ids': torch.tensor([], device=device),
    }
