from pathlib import Path
import torch
import datetime

# 获取当前日期和时间
now = datetime.datetime.now()
# 格式化日期时间为 "月日时分" 格式
formatted_time = now.strftime("%m%d%H%M")

# 配置类
class Config:
    DEBUG: bool = False

    # XVLM模型相关配置
    model_name: Path = Path("xvlm/xvlm-base")
    xvlm_config = {
        'embed_dim': 1024,  # 根据需要调整
        'temp': 0.07,       # 对比学习的温度参数
        # 其他XVLM相关的配置参数
    }

    MODEL_NAME: str = f"{model_name}"
    EXP_FOLDER: Path = Path(f"exp_logs/log_{formatted_time}/{MODEL_NAME}/exp0")
    LOG_DIR: Path = Path(f"exp_logs/log_{formatted_time}/{MODEL_NAME}/logs")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

    EPOCHS: int = 40
    BATCH_SIZE: int = 6  # 根据显存调整
    VAL_BATCH_SIZE: int = 6
    SAVE_INTERVAL: int = 4  # 每隔多少个epoch保存一次检查点

    IMG_SIZE: int = 384  # 384

    NUM_WORKERS: int = 4
    VAL_NUM_WORKERS: int = 4

    #损失函数参数
    ALPHA_INITIAL: float = 0.5  # 初始alpha值
    ALPHA_FINAL: float = 0.8  # 最终alpha值
    belta_initial: float = 0.4  # 初始belta
    belta_final: float = 0.7  # 最终belta

    #优化器参数、学习率调度器参数
    # LEARNING_RATE: float = 0.0001  # 学习率
    # WEIGHT_DECAY: float = 0.01  # 权重衰减
    # LR_MULT: float = 2.0  # 学习率倍增因子
    # eps: float = 1e-8  # AdamW的epsilon
    # betas: tuple = (0.9, 0.98)  # AdamW的betas
    # warmup_epochs: float = 0.1  # 预热阶段的epoch数
    # SCHEDULER: str = 'linear'  # 调度器类型，可扩展



