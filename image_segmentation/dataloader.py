import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from PIL import Image

class OxfordPetsDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='trainval'):
        self.dataset = OxfordIIITPet(root=root, split=split, target_types='segmentation', download=True)

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

        # Oxford Pets 原本的mask中
        # 0 = 背景
        # 1 = 宠物
        # 2 = 边界
        mask = torch.from_numpy(mask).long()

        return img, mask

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2):
    # Oxford Pets官方没有官方val/test划分，我们用trainval
    dataset = OxfordPetsDataset(root=root, split='trainval')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader



