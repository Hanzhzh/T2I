import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.nn import functional as F
from ruamel.yaml import YAML

from model.bench import Bench
from model.tokenization_bert import BertTokenizer
from model.tokenization_roberta import RobertaTokenizer
from dataset.dataloader import custom_collate_fn
import utils


def setup_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现。
    """
    print(f"Setting random seed to {seed}")
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_config(config_path):
    """
    加载 YAML 配置文件。
    """
    yaml = YAML(typ='rt')
    with open(config_path, 'r') as f:
        return yaml.load(f)


def parse_gpus(gpu_str):
    """
    解析GPU字符串，返回GPU列表和GPU数量。
    """
    gpus = [int(x) for x in gpu_str.split(',') if x.strip().isdigit()]
    return gpus, len(gpus)


def load_model(checkpoint_path, config, device, num_gpus, use_roberta):
    """
    加载模型及其权重。
    """
    print(f"Loading model from {checkpoint_path}")
    model = Bench(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model']

    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=config['GPUS'])
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    model = model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    return model


def get_tokenizer(config):
    """
    根据配置选择并返回相应的分词器。
    """
    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    return tokenizer


def encode_text(model, tokenizer, text, device, config):
    """
    对文本进行编码，返回文本嵌入。
    """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(inputs)  # 确保 Bench 模型有 encode_text 方法
    return text_embeddings

def encode_image(model, image_path, device):
    """
    对图像进行编码，返回图像嵌入。
    """
    image = Image.open(image_path).convert('RGB')
    # 假设有一个预处理函数，根据您的模型调整
    preprocess = utils.get_preprocess()
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image(image_tensor)  # 确保 Bench 模型有 encode_image 方法
    return image_embeddings


def compute_similarity(image_emb, text_emb):
    """
    计算图像嵌入和文本嵌入之间的余弦相似度。
    """
    image_emb = F.normalize(image_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    similarity = torch.matmul(image_emb, text_emb.transpose(0, 1))
    return similarity.item()


def main(args, config):
    """
    主函数，处理编码和相似度计算。
    """
    setup_seed(config.get('seed', 42))

    # 解析GPU配置
    gpus, num_gpus = parse_gpus(config['GPUS'])
    print(f"Using GPUs: {gpus} (Total: {num_gpus})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = config['GPUS']

    # 获取分词器
    tokenizer = get_tokenizer(config)

    # 准备图像和文本
    image_path = args.image_path
    text = args.text
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 获取所有检查点
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # 存储相似度结果
    similarity_results = {}

    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        model = load_model(ckpt_path, config, device, num_gpus, config['use_roberta'])

        # 编码图像和文本
        image_emb = encode_image(model, image_path, device)
        text_emb = encode_text(tokenizer, text, device, config)

        # 计算相似度
        similarity = compute_similarity(image_emb, text_emb)
        similarity_results[ckpt] = similarity
        print(f"Checkpoint: {ckpt} | Similarity: {similarity:.4f}")

    # 保存结果到JSON文件
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(similarity_results, f, indent=4)
    print(f"Similarity results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="评估相似度")
    parser.add_argument('--config', default='i2t.yaml', type=str, help="Path to the YAML configuration file.")
    parser.add_argument('--checkpoint_dir', default='../output/train',required=True, type=str,
                        help="Directory containing model checkpoints (.pth files).")
    parser.add_argument('--image_path', required=True, type=str, help="Path to the input image.")
    parser.add_argument('--text', required=True, type=str, help="Description text corresponding to the image.")
    parser.add_argument('--output', default='similarity_results.json', type=str,
                        help="Path to save the similarity results JSON.")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确保输出目录存在
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    main(args, config)
