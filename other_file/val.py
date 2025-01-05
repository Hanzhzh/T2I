import os
import json
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
# from config import processor

def validate(model, dataloader, device, writer, epoch: int):
    model.eval()
    correct_1 = correct_5 = correct_10 = 0
    total = 0
    total_rank_percentage = 0.0  # 用于计算平均排名百分比

    drone_image_features = []
    drone_text_features = []
    drone_labels = []
    drone_image_paths = []  # 初始化无人机图像路径列表

    satellite_image_features = []
    satellite_text_features = []
    satellite_labels = []
    satellite_image_paths = []

    # 用于记录 Recall@5 的匹配结果
    recall5_matches = []
    # 保存 Recall@5 匹配结果到 JSON 文件
    output_dir = "/recall@5_file"
    output_file = f"{output_dir}/recall5_matches_epoch_{epoch + 1}.json"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.info(f"已存在目录 '{output_dir}'，已清空其内容")
        # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"目录 '{output_dir}' 已创建")

    def get_label_from_path(image_path):
        parts = Path(image_path).parts
        if 'drone' in parts:
            index = parts.index('drone')
        elif 'satellite' in parts:
            index = parts.index('satellite')
        else:
            return None
        return parts[index + 1] if index + 1 < len(parts) else None

    # print("加载 DINOv2 图像预处理器...")
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # checkpoint_path = '/home/hanzz/projects/image_caption/clip/Long-CLIP/exp_logs/log_12071019/dinov2/exp0/ckpts/best_model.pth'
    # # 加载完整检查点
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # # 加载模型的 state_dict
    # model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for batch_idx, (images, text_embeddings, image_paths, subset_ids, sample_indices) in enumerate(
                tqdm(dataloader, desc="Validating")):
            if len(images) == 0:
                print(f"警告: 批次 {batch_idx} 中没有图像数据。")
                continue

            images = images.to(device).float()
            text_feats = text_embeddings.to(device)

            # inputs = processor(images=images, return_tensors="pt").to(device)
            # outputs = model(**inputs)
            # image_feats = outputs.pooler_output.float()
            # # 图像预处理
            # transform_1 = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
            #                          std=(0.26862954, 0.26130258, 0.27577711))
            # ])
            # image_feats = transform_1(image_feats)
            # print(image_feats)

            # 前向传播
            outputs = model(images, text_feats)
            image_features, text_features = outputs[:2]

            for i in range(len(image_paths)):
                image_path = image_paths[i]
                label = get_label_from_path(image_path)

                if 'drone' in image_path:
                    drone_image_features.append(image_features[i].cpu())
                    drone_text_features.append(text_features[i].cpu())
                    drone_labels.append(label)
                    drone_image_paths.append(image_path)  # 收集无人机图像路径
                elif 'satellite' in image_path:
                    satellite_image_features.append(image_features[i].cpu())
                    satellite_text_features.append(text_features[i].cpu())
                    satellite_labels.append(label)
                    satellite_image_paths.append(image_path)
                else:
                    print(f"未知的路径: {image_path}")

    if len(satellite_image_features) == 0 or len(drone_text_features) == 0:
        print("验证数据不足，无法计算准确率。")
        return 0.0  # 确保返回 float

    # 创建标签到卫星图像索引的映射
    label_to_satellite = {}
    for idx, label in enumerate(satellite_labels):
        if label not in label_to_satellite:
            label_to_satellite[label] = []
        label_to_satellite[label].append(idx)

    satellite_image_features = torch.stack(satellite_image_features)
    drone_text_features = torch.stack(drone_text_features)

    # 归一化特征向量
    satellite_image_features = F.normalize(satellite_image_features, p=2, dim=1)
    drone_text_features = F.normalize(drone_text_features, p=2, dim=1)

    # 预计算所有相似度
    similarities_matrix = torch.matmul(drone_text_features, satellite_image_features.t())  # [num_drone, num_satellite]
    sorted_similarities, sorted_indices = torch.sort(similarities_matrix, dim=1, descending=True)

    num_satellites = satellite_image_features.size(0)
    print(f"Total number of satellite images: {num_satellites}")

    for i in range(len(drone_text_features)):
        true_label = drone_labels[i]
        # 获取真实匹配图像的索引
        true_indices = label_to_satellite.get(true_label, [])
        if not true_indices:
            print(f"警告: 无法找到标签为 {true_label} 的卫星图像。")
            continue  # 跳过没有真实匹配的情况

        # 假设每个标签只有一个真实匹配图像，如果有多个，可以根据需要调整
        true_idx = true_indices[0]
        true_similarity = similarities_matrix[i, true_idx].item()
        row = similarities_matrix[i]
        label_sort = 0
        for j in range(len(row)):
            if row[j] < true_similarity:
                label_sort += 1
        per = label_sort / len(row)
        true_image_path = satellite_image_paths[true_idx]
        print(
            f"无人机文本真实标签{true_label},大小百分比为{per}, 对应的真实卫星图像路径为：{true_image_path}, 真实相似度为{true_similarity}")

        true_image_path = satellite_image_paths[true_idx]

        # 获取真实匹配图像在排序中的位置（排名）
        # 注意：sorted_indices 按照 descending=True 排序
        # 因此，排名从1开始
        rank = (sorted_indices[i] == true_idx).nonzero(as_tuple=True)[0].item() + 1  # 排名位置
        rank_percentage = (rank / num_satellites) * 100  # 百分比表示

        # 获取 Recall@5 的检索结果
        retrieved_indices = sorted_indices[i][:5]
        retrieved_paths = [satellite_image_paths[idx] for idx in retrieved_indices]
        retrieved_similarities = sorted_similarities[i][:5].cpu().tolist()
        retrieved_labels = [satellite_labels[idx] for idx in retrieved_indices]

        # 计算 Recall@1, Recall@5, Recall@10
        top_1_label = satellite_labels[sorted_indices[i][0]]
        top_5_labels = retrieved_labels[:5]
        top_10_labels = [satellite_labels[idx] for idx in sorted_indices[i][:10]]

        total += 1
        if true_label == top_1_label:
            correct_1 += 1
        if true_label in top_5_labels:
            correct_5 += 1
        if true_label in top_10_labels:
            correct_10 += 1

        # 累加排名百分比
        total_rank_percentage += rank_percentage

        # 记录 Recall@5 的匹配结果
        recall5_matches.append({
            "query_image_path": drone_image_paths[i],  # 使用无人机图像路径
            "retrieved_image_paths": retrieved_paths,
            "retrieved_similarities": retrieved_similarities,  # 添加相似性分数
            "true_label": true_label,
            "retrieved_labels": top_5_labels,
            "true_match_image_path": true_image_path,  # 添加真实匹配图像路径
            "true_match_similarity": true_similarity,  # 添加真实匹配相似性分数
            "true_match_rank_position": rank,  # 添加真实匹配的排名位置
            "true_match_rank_percentage": f"{rank_percentage}%"  # 添加排名百分比
        })

    recall_1 = correct_1 / total if total > 0 else 0.0
    recall_5 = correct_5 / total if total > 0 else 0.0
    recall_10 = correct_10 / total if total > 0 else 0.0
    average_rank_percentage = total_rank_percentage / total if total > 0 else 0.0

    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"Average True Match Rank Percentage: {average_rank_percentage:.2f}%")

    # 保存 JSON 文件
    with open(output_file, "w") as f:
        json.dump(recall5_matches, f, indent=4)
    print(f"Recall@5 匹配结果数量: {len(recall5_matches)}")
    print(f"Recall@5 匹配结果已保存到 {output_file}")

    # 记录到 TensorBoard
    writer.add_scalar('Recall/Recall@1', recall_1, epoch + 1)
    writer.add_scalar('Recall/Recall@5', recall_5, epoch + 1)
    writer.add_scalar('Recall/Recall@10', recall_10, epoch + 1)
    writer.add_scalar('Recall/Average_True_Match_Rank_Percentage', average_rank_percentage, epoch + 1)

    return recall_10  # 确保返回 float
