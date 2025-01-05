import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image, ImageFile
from pathlib import Path
from typing import Optional
from torch.utils.data import ConcatDataset

from .randaugment import RandomAugment

# 允许加载截断的图像和取消图像大小限制
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

"""
使用的数据集格式：
[
    {
        "image_id": "img1",
        "image": "images/img1.jpg",
        "Description 1": "A beautiful sunrise over the mountains.",
        "Description 2": "The sky is painted with hues of orange and pink."
    },
    {
        "image_id": "img2",
        "image": "images/img2.jpg",
        "Description 1": "A serene lake surrounded by forests.",
        "Description 2": "The water reflects the clear blue sky."
    }
    // 更多数据...
]

"""

class CustomImageTextDataset(Dataset):
    """自定义数据集类，用于加载图像和相应的文本，每个图像有一个对应的npy文件
    没有将文本提前转为编码
    """

    def __init__(self, config, image_folder: str, desc_folder: str, transform: Optional[transforms.Compose] = None ):
        self.image_folder = Path(image_folder)
        self.desc_folder = Path(desc_folder)
        self.transform = transform
        self.config = config
        # 定义输出文件为实例属性
        self.output_file = Path("../output/dataset.json")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)  # 确保路径存在

        self.image_paths = []
        print("初始化~~~CustomImageTextDataset~~~")

        # 遍历所有主要文件夹
        for main_folder in self.config['view_folder']:
            main_image_folder = self.image_folder / main_folder
            main_des_folder = self.desc_folder / f"{main_folder}_json"
            print(f"正在处理主要文件夹: {main_folder}")

            if not main_image_folder.exists():
                print(f"主要图像文件夹不存在：{main_image_folder}")
                continue

            # 遍历所有子文件夹
            for subset in sorted(main_image_folder.iterdir()):
                for image_file in sorted(subset.iterdir()):
                    if image_file.is_file() and image_file.suffix in ['.jpeg', '.jpg', '.png']:
                        # 获取对应的npy文件路径
                        # npy_file = main_des_folder / subset.name / f"{image_file.stem}.npy"
                        json_file = main_des_folder / subset.name /f"{subset.name}.json"
                        self.image_paths.append((image_file, json_file))
                        # print(f"图文对的路径为：\n 图像路径{image_file} \n 文本路径{npy_file}")

        # 一次性写入 JSON 文件
        with open(self.output_file, "w") as file:
            json.dump(
                [{"image_path": str(p[0]), "json_path": str(p[1])} for p in self.image_paths],
                file,
                indent=4
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, json_path = self.image_paths[idx]

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图像，替换为空白图像 {image_path}: {e}")
            image = Image.new('RGB', (self.config['image_res'], self.config['image_res']))

        if self.transform:
            image = self.transform(image)

        # 加载对应的npy文件（假设包含文本字符串）
        subset_id = str(image_path.parent.relative_to(image_path.parent.parent.parent))  # 例如：'view/subfolder'
        sample_index = idx  # 样本索引
        try:
            # 读取 JSON 文件
            with open(json_path, 'r') as file:
                data = json.load(file)
            desc_dict= data.get(f"{image_path.stem}{image_path.suffix}", {})

            if isinstance(desc_dict, dict):
                # 将描述字典封装在一个新的字典中，以图像ID为键
                desc_text = [{f"{subset_id}/{image_path.stem}{image_path.suffix}": desc_dict}]
            else:
                desc_text = [{f"{image_path.stem}{image_path.suffix}": "没有在json文件中找到对应的图像描述"}]

        except Exception as e:
            print(f"无法加载json文件 {json_path}: {e}, 使用空文本")
            desc_text = ""

        # return image, desc_text, str(image_path), str(json_path), subset_id, sample_index
        return image, desc_text, subset_id


def custom_collate_fn(batch):
    images, desc_texts, subset_ids = zip(*batch)

    # 保持 desc_texts 为列表的列表
    desc_texts = list(desc_texts)

    return (
        torch.stack(images, 0),
        desc_texts,
        list(subset_ids)
    )


def create_dataset(dataset_type, config, evaluate=False):
    """
        输入：
        dataset_type：数据集类型（如 I2T_university），类型为 str。
        config：包含配置参数（如图像大小、文件路径等）的字典，类型为 dict。
        evaluate：布尔值，表示是否仅创建测试集，类型为 bool。

        输出：
        如果 evaluate 为 True，返回 None 和测试数据集；
        否则，返回训练数据集和测试数据集。

        逻辑：
        根据传入的 dataset_type 判断数据集类型。如果是 I2T_university：
        创建测试数据集的变换（test_transform），并生成 EvalImageDataset 实例。
        如果 evaluate 为 True，只返回测试数据集。
        否则，创建训练数据集的变换（train_transform），并生成 ImageTextDataset 实例。
        可以扩展以支持更多数据集类型。
    """
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )

    if dataset_type == 'I2T_university':
        # 测试集变换
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        # 加载验证集
        test_dataset = CustomImageTextDataset(
            image_folder=str(Path(config['data_folder']) / "test"),
            desc_folder=str(Path(config['data_folder']) / "test_json"),
            transform=test_transform,
            config=config
        )
        print("验证数据加载完成", flush=True)

        if evaluate:
            return None, test_dataset

        # 训练集变换
        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=['Identity', 'Equalize', 'Brightness', 'Sharpness', 'Rotate', 'Rotate']
            ),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset_1 = CustomImageTextDataset(
            image_folder=str(Path(config['data_folder']) / "train"),
            desc_folder=str(Path(config['data_folder']) / "train_json"),
            transform=train_transform,
            config=config
        )

        train_dataset_2 = CustomImageTextDataset(
            image_folder=str(Path(config['data_folder']) / "train"),
            desc_folder=str(Path(config['data_folder']) / "train_json"),
            transform=test_transform,
            config=config
        )
        print("训练数据加载完成", flush=True)
        print(f"合并前的训练集包含 {len(train_dataset_2)} 个样本", flush=True)
        # 合并数据集
        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
        # 输出合并后的数据集长度
        print(f"合并后的训练集包含 {len(train_dataset)} 个样本, 应该是合并前的2倍", flush=True)
        # 获取第一个样本

        sample = train_dataset[0]

        # 原始图像
        original_image = sample[0]
        # 图像描述
        desc_text = sample[1]
        # 子文件夹 ID
        subset_id = sample[2]
        # 打印内容
        print("训练集第一个样本：", flush=True)
        print("原始图像类型:", type(original_image), flush=True)  # PIL.Image
        print("图像描述:", desc_text, flush=True)
        print("子文件夹 ID:", subset_id, flush=True)

        return train_dataset, test_dataset

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders