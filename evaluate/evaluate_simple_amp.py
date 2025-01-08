import torch
import torch.nn.functional as F
import gc
import time
import datetime
from collections import defaultdict
from tqdm import tqdm


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
