from typing import Tuple
from pathlib import Path
import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import load_augumented_data_as_list, tokenize_img

classes_int2str = {
    0: '1m',
    1: '2m',
    2: '3m',
    3: '4m',
    4: '5m',
    5: '6m',
    6: '7m',
    7: '8m',
    8: '9m',
    9: '1p',
    10: '2p',
    11: '3p',
    12: '4p',
    13: '5p',
    14: '6p',
    15: '7p',
    16: '8p',
    17: '9p',
    18: '1s',
    19: '2s',
    20: '3s',
    21: '4s',
    22: '5s',
    23: '6s',
    24: '7s',
    25: '8s',
    26: '9s',
    27: '1z',
    28: '2z',
    29: '3z',
    30: '4z',
    31: '5z',
    32: '6z',
    33: '7z',
    34: '0m',
    35: '0p',
    36: '0s',
    # 37: 'back'  # 牌背面
}

classes_str2int = {}

for k, v in classes_int2str.items():
    classes_str2int[v] = k

# 定义自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, data_list: list[Tuple[str, np.ndarray]], transform=None):
        for p, i in enumerate(data_list):
            data_list[p] = (classes_str2int[i[0]], torch.from_numpy(i[1]).float())
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label, image = self.data_list[idx]
        # 将图像数据转换为PyTorch张量
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_loader(batch_size=16, path = None, cache_path = None):
    path = path or Path('Assets/label')
    cache_path = cache_path or Path('Assets/aug_dataset.cache')
    if cache_path.exists():
        dataset = torch.load(cache_path)
    else:
        data_list = load_augumented_data_as_list(path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # ... 这里可以加入其他的变换，比如数据标准化等
        ])
        dataset = CustomDataset(data_list)
        torch.save(dataset, cache_path)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=1)

def get_test_loader(batch_size=16, path=None):
    path = path or Path('Assets/unlabel/testset')
    ds = []
    for label in os.listdir(path):
        for fn in os.listdir(path / label):
            tokenized = tokenize_img(str((path / label / fn).absolute()))
            ds.append((label, tokenized))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # ... 这里可以加入其他的变换，比如数据标准化等
    ])
    dataset = CustomDataset(ds)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
