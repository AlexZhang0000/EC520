import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.ade20k import ADE20K
import torchvision.transforms as T
from PIL import Image

# 类别映射
ADE20K_VALID_CLASSES = {
    2: 1,   # building
    12: 2,  # tree
    14: 3   # person
}
BACKGROUND_LABEL = 0

class ADE20KSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='train'):
        self.dataset = ADE20K(root=root, split=mode, target_types='segmentation', download=True)

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

        # 重新映射标签
        new_mask = np.ones_like(mask) * BACKGROUND_LABEL
        for orig_class, new_class in ADE20K_VALID_CLASSES.items():
            new_mask[mask == orig_class] = new_class

        mask = torch.from_numpy(new_mask).long()

        return img, mask

def get_loader(root, batch_size=8, mode='train', shuffle=True, num_workers=2):
    dataset = ADE20KSubsetDataset(root=root, mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader




