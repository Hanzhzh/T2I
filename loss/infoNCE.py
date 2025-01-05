import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict

class InfoNCE_Loss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        计算标准的 InfoNCE 损失。

        参数:
        - image_features: 图像特征 [batch_size, feature_dim]
        - text_features: 文本特征 [batch_size, feature_dim]

        返回:
        - loss: InfoNCE 损失标量
        """
        # 特征归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # 检查 logits 是否包含 NaN 或 Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("警告：logits 包含 NaN 或 Inf")

        # 创建标签，假设对角线为正样本
        labels = torch.arange(logits.size(0)).to(logits.device)

        # 计算交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        # 检查损失是否为 NaN
        if torch.isnan(loss):
            print("警告：损失值为 NaN")

        return loss