import torch
import torch.nn.functional as F
import gc
import time
import datetime
from collections import defaultdict
from tqdm import tqdm

import utils


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
