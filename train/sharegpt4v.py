import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

from randaugment import RandomAugment

def count_ids_in_json_file(json_file_path, count_unique=False):
    """
    统计单个 JSON 文件中的 'id' 数量。
    :param json_file_path: JSON 文件路径
    :param count_unique: 是否统计唯一 'id' 数量，默认为 False（统计总数）
    :return: id 的数量
    """
    try:
        # 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):  # 确保 JSON 文件是一个列表
                # 提取所有 'id'
                ids = [item["id"] for item in data if "id" in item]
                if count_unique:
                    unique_ids = set(ids)
                    print(f"文件 {os.path.basename(json_file_path)} 中共有 {len(unique_ids)} 个唯一 id")
                    return len(unique_ids)
                else:
                    print(f"文件 {os.path.basename(json_file_path)} 中共有 {len(ids)} 个 id（包括重复）")
                    return len(ids)
            else:
                print(f"文件 {json_file_path} 不是预期的列表结构，跳过")
                return 0
    except Exception as e:
        print(f"读取文件失败: {json_file_path}, 错误: {e}")
        return 0

# data4v_root = 'sharegpt4v/data/'
# json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
# image_root = 'sharegpt4v/data/'

data4v_root = '/home/hanzz/data_file/U1652-all/'
json_name = "train_json/U1652.json"
image_root = '/home/hanzz/data_file/U1652-all/'

# drone_name = "train_json/drone_roof.json"
# satellite_name = "train_json/satellite_roof.json"

# 调试并统计
total_id_count = count_ids_in_json_file(data4v_root + json_name, count_unique=False)
unique_id_count = count_ids_in_json_file(data4v_root + json_name, count_unique=True)
print(f"数据集总数量为：{total_id_count}")
# print(f"satellite数据集总数量为：{unique_id_count}")
ratio = 0.2

class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        # self.total_len = 1000
        self.total_len = int(total_id_count * ratio)
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _ , self.preprocess = clip.load("ViT-L/14")
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # caption = self.json_data[index]['conversations'][1]['value']
        roof = self.json_data[index]['caption']["roof"]
        roof = roof.replace("\n", " ")
        around_1 = self.json_data[index]['caption']["around_1"]
        around_1 = around_1.replace("\n", " ")
        around_2  = self.json_data[index]['caption']["around_2"]
        around_2 = around_2 .replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, roof, around_1, around_2


class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = int(unique_id_count * ratio)
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len:]
        _ , self.preprocess = clip.load("ViT-L/14")

        # # 初始化数据增强
        self.augmenter = RandomAugment(N=2, M=10, isPIL=True,
                                       augs=['Identity', 'Equalize', 'Brightness', 'Sharpness', 'Rotate'])

        print("---------------------------------")
        print(f"{self.augmenter.augs}")
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        roof = self.json_data[index]['caption']["roof"]
        roof = roof.replace("\n", " ")
        around_1 = self.json_data[index]['caption']["around_1"]
        around_1 = around_1.replace("\n", " ")
        around_2  = self.json_data[index]['caption']["around_2"]
        around_2 = around_2 .replace("\n", " ")
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        
        # id_ = self.json_data[index]['id']

        # 数据增强
        image = self.augmenter(image)  # 对图像应用增强
        
        # 如果图像是 numpy.ndarray，转换为 PIL.Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 将图像转换为预期的 tensor 格式
        image_tensor = self.preprocess(image)

        return image_tensor, roof, around_1, around_2
