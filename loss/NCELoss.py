import torch
import torch.nn.functional as F
from collections import defaultdict

class WeightedInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.1, init_alpha=0.7, init_beta=0.5):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.alpha = torch.nn.Parameter(torch.tensor(init_alpha))
        self.beta = torch.nn.Parameter(torch.tensor(init_beta))

    def forward(self, image_features, text_features, subset_ids):
        """
        前向计算函数

        参数说明：
        - image_features: 图像特征 [batch_size, feature_dim]
        - text_features: 文本特征 [batch_size, feature_dim]
        - subset_ids: 子集ID列表 [batch_size]，形如 "viewA/123"

        返回值：
        - loss: 最终计算得到的加权InfoNCE损失标量。
        """

        # # 特征归一化
        # image_features = F.normalize(image_features, dim=-1)
        # text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = torch.matmul(text_features, image_features.t()) / self.temperature
        # print("损失函数的相似度矩阵：")
        # print(f"logits_per_image:{logits_per_image}")

        # 获取batch size
        batch_size = len(subset_ids)
        if batch_size <= 1:
            raise ValueError("Batch size must be greater than 1 to compute InfoNCE loss.")

        # 初始化权重矩阵，非正样本对的初始值为 label_smoothing/(batch_size-1)
        weight_matrix = torch.full((batch_size, batch_size),
                                   fill_value=self.label_smoothing / (batch_size - 1),
                                   device=image_features.device)

        # 解析subset_ids，分离view与id
        try:
            views, ids, name = zip(*[sid.split("/") for sid in subset_ids])
        except ValueError as e:
            raise ValueError("每个 subset_id 应包含一个 '/' 用于分离 view 和 id。") from e

        views = list(views)
        ids = list(ids)
        name = list(name)

        # 使用字典将相同id的索引分组，方便处理同一个子集中的样本对
        id_to_indices = defaultdict(list)
        for idx, id_ in enumerate(ids):
            id_to_indices[id_].append(idx)

        # 为每个子集id分配权重
        for id_, indices in id_to_indices.items():
            if len(indices) == 1:
                # 若某id下只有一个样本，无法形成正样本对，跳过
                continue
            indices = torch.tensor(indices, device=image_features.device)
            # 获取这些样本的view信息
            sample_views = [views[i] for i in indices.tolist()]

            # 假设"view1"为一种特定视图，其余为另一视图
            sample_views_bool = torch.tensor([v == 'view1' for v in sample_views], device=image_features.device)

            # 构建view相同与否的掩码矩阵
            view_matrix = sample_views_bool.unsqueeze(1) == sample_views_bool.unsqueeze(0)  # [n,n]

            # 主正样本对：对角线位置(自身匹配)，权重为 1.0 - label_smoothing
            weight_matrix[indices, indices] = 1.0 - self.label_smoothing

            # 次级正样本对（同id但不同样本）权重设定
            # 将所有正样本对(同id)的位置区分为两种情况：同view和不同view
            same_view_mask = view_matrix.clone()
            same_view_mask.fill_diagonal_(False)  # 对角线是主对，已处理过

            # 不同view的mask（在同一个id子集下的非对角线位置）
            different_view_mask = ~same_view_mask & (torch.ones_like(same_view_mask, dtype=torch.bool))

            # 将同id的非对角线位置根据view类型赋予权重
            # 同view的正样本对：alpha * (1.0 - label_smoothing)
            # 不同view的正样本对：beta * (1.0 - label_smoothing)
            weight_matrix[indices.unsqueeze(1), indices.unsqueeze(0)] = (
                    self.alpha * (1.0 - self.label_smoothing) * same_view_mask.float() +
                    self.beta * (1.0 - self.label_smoothing) * different_view_mask.float()
            )

        # 再次确保对角线(主正样本对)正确设置
        diag_indices = torch.arange(batch_size, device=image_features.device)
        weight_matrix[diag_indices, diag_indices] = 1.0 - self.label_smoothing

        # 计算图像到文本方向的损失
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, weight_matrix)

        # 计算文本到图像方向的损失
        loss_t = F.binary_cross_entropy_with_logits(logits_per_text, weight_matrix.t())

        # 最终损失是两个方向的平均
        loss = (loss_i + loss_t) / 2

        return loss
