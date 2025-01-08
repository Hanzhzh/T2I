import torch
import torch.nn.functional as F
import gc
import time
import datetime
from collections import defaultdict
from tqdm import tqdm

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
