import os
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

# 选定的VOC类别索引
VOC_TARGET_CLASSES = {
    15: 1,  # person
    8: 2,   # cat
    12: 3,  # dog
    7: 4    # car
}
# 其他类别都设为 ignore_index=255

class VOCSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', download=True):
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)

        self.input_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
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

        # 重新映射mask
        remapped_mask = np.full_like(mask, fill_value=255)  # 默认ignore
        for voc_class, mapped_class in VOC_TARGET_CLASSES.items():
            remapped_mask[mask == voc_class] = mapped_class
        remapped_mask[mask == 0] = 0  # 背景保留为0

        remapped_mask = torch.from_numpy(remapped_mask).long()

        return img, remapped_mask

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2):
    dataset = VOCSubsetDataset(root=root, image_set='train' if mode=='train' else 'val', download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader





