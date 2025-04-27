import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from PIL import Image

# VOC官方定义的21个类别顺序
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 我们选的4个类别的原始ID
CLASS_NAME_TO_ID = {
    'background': 0,
    'sky': 21,    # 注意：天空不是官方VOC标签，我们特殊处理（全归到背景也可以）
    'person': 15,
    'car': 7,
}

# 如果找不到天空类别（因为VOC标准分割没有"sky"），就用背景代替

# 只保留背景、人、车，把其他当成背景
ID_MAPPING = {i: 0 for i in range(256)}
ID_MAPPING.update({
    0: 0,    # background
    15: 2,   # person
    7: 3,    # car
    # (天空单独处理)
})

class VOC4ClassDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train'):
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=True)

        self.input_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        img = self.input_transform(img)

        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask)

        # 重映射标签：只保留背景、人、车，其它都归0
        mask_remapped = np.zeros_like(mask)
        for k, v in ID_MAPPING.items():
            mask_remapped[mask == k] = v

        mask_remapped = torch.from_numpy(mask_remapped).long()

        return img, mask_remapped

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2):
    dataset = VOC4ClassDataset(root=root, image_set=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader



