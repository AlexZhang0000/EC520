import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

class OxfordPetsDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='train', distortion=None):
        full_dataset = OxfordIIITPet(root=root, split='trainval', target_types='segmentation', download=True)
        self.distortion = distortion

        # 手动划分 train/test
        idx_list = list(range(len(full_dataset)))
        train_idx, test_idx = train_test_split(idx_list, test_size=0.2, random_state=42)

        if mode == 'train':
            self.selected_idx = train_idx
        elif mode == 'val' or mode == 'test':
            self.selected_idx = test_idx
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'")

        self.dataset = full_dataset

        self.input_transform_list = [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
        ]

        self.input_transform_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]

        self.input_transform = T.Compose(self.input_transform_list)

    def __len__(self):
        return len(self.selected_idx)

    def __getitem__(self, idx):
        real_idx = self.selected_idx[idx]
        img, mask = self.dataset[real_idx]

        img = self.input_transform(img)

        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()

        return img, mask

def get_loader(root, batch_size=8, mode='train', shuffle=True, num_workers=2, distortion=None):
    dataset = OxfordPetsDataset(root=root, mode=mode, distortion=distortion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader




