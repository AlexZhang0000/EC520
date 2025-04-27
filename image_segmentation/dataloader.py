import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', download=True):
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)

        self.input_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ✅加标准归一化
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        img = self.input_transform(img)

        target = target.resize((256, 256), Image.NEAREST)
        target = np.array(target)
        target = torch.from_numpy(target).long()

        return img, target

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2):
    dataset = VOCDataset(root=root, image_set=mode, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader



