import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from pathlib import Path
from typing import Optional

from other_file.config import Config
from dataset.randaugment import RandomAugment

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

    def __init__(self, image_folder: str, desc_folder: str, transform: Optional[transforms.Compose] = None):
        self.image_folder = Path(image_folder)
        self.desc_folder = Path(desc_folder)
        self.transform = transform
        # 定义输出文件为实例属性
        self.output_file = Path("./output/logs/dataset.json")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)  # 确保路径存在

        self.image_paths = []
        print("初始化~~~CustomImageTextDataset~~~")

        # 遍历所有主要文件夹
        for main_folder in Config.view_folder:
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
                [{"image_path": str(p[0]), "npy_path": str(p[1])} for p in self.image_paths],
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
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        # 加载对应的npy文件（假设包含文本字符串）
        subset_id = str(image_path.parent.relative_to(image_path.parent.parent.parent))  # 例如：'view/subfolder'
        sample_index = idx  # 样本索引
        try:
            # 读取 JSON 文件
            with open(json_path, 'r') as file:
                data = json.load(file)
            desc_dict = data.get(f"{image_path.stem}{image_path.suffix}", {})

            if isinstance(desc_dict, dict):
                # 将描述字典封装在一个新的字典中，以图像ID为键
                desc_text = [{f"{image_path.stem}{image_path.suffix}": desc_dict}]
            else:
                desc_text = {f"{image_path.stem}{image_path.suffix}": "没有在json文件中找到对应的图像描述"}

            print(desc_text)

            # 检查是否成功获取到该图像的描述
            # if isinstance(description, dict):
            # description_1 = description.get('Description 1', '没有在json文件中找到Description 1描述')
            # description_2 = description.get('Description 2', '没有在json文件中找到Description 2描述')
        except Exception as e:
            print(f"无法加载json文件 {json_path}: {e}, 使用空文本")
            desc_text = ""

        # desc = {f"{subset_id}/{image_path.stem}{image_path.suffix}": desc_text}

        return image, desc_text, str(image_path), str(json_path), subset_id, sample_index


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle)
        samplers.append(sampler)
    return samplers

def custom_collate_fn(batch):
    images, desc_texts, image_paths, json_paths, subset_ids, sample_indices = zip(*batch)

    # 保持 desc_texts 为列表的列表
    desc_texts = list(desc_texts)

    return (
        torch.stack(images, 0),
        desc_texts,
        list(image_paths),
        list(json_paths),
        list(subset_ids),
        list(sample_indices)
    )

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


def create_dataset(dataset_type, evaluate=False):
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
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        # 加载验证集
        test_dataset = CustomImageTextDataset(
            image_folder=str(Config.DATA_FOLDER / "test"),
            desc_folder=str(Config.VAL_NPY_FOLDER),
            transform=test_transform,
        )
        print("验证数据加载完成")

        if evaluate:
            return None, test_dataset

        # 训练集变换
        train_transform = transforms.Compose([
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=['Identity', 'Equalize', 'Brightness', 'Sharpness', 'Rotate', 'Rotate']
            ),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = CustomImageTextDataset(
            image_folder=str(Config.DATA_FOLDER / "train"),
            desc_folder=str(Config.TRAIN_NPY_FOLDER),
            transform=train_transform,
        )
        print("训练数据加载完成")

        return train_dataset, test_dataset


import json
import pickle


# 保存数据集为 JSON 文件
def save_dataset_to_json(dataset, file_path):
    dataset_list = []
    for i in range(len(dataset)):
        # 调用 __getitem__ 获取单条数据
        image, desc_text, image_path, json_path, subset_id, sample_index = dataset[i]
        dataset_list.append({
            "image_path": image_path,
            "desc_text": desc_text,
            "json_path": json_path,
            "subset_id": subset_id,
            "sample_index": sample_index
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)


# 保存数据集为 Pickle 文件
def save_dataset_to_pickle(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)


# 加载数据集从 Pickle 文件
def load_dataset_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

train_data, test_data = create_dataset('I2T_university')  # 创建训练和测试数据集

# 保存训练集和测试集
save_dataset_to_json(train_data, "output/logs/train_dataset.json")
save_dataset_to_json(test_data, "output/logs/test_dataset.json")

save_dataset_to_pickle(train_data, "output/logs/train_dataset.pkl")
save_dataset_to_pickle(test_data, "output/logs/test_dataset.pkl")


