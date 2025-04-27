import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data = []
        self.labels = []
        self.mode = mode

        if mode == 'train':
            # 读取 data_batch_1 ~ data_batch_4
            for i in range(1, 5):
                batch_file = os.path.join(data_path, f'data_batch_{i}')
                self._load_batch(batch_file)
        elif mode == 'val':
            # 读取 validation_batch
            batch_file = os.path.join(data_path, 'validation_batch')
            self._load_batch(batch_file)
        elif mode == 'test':
            # 读取 test_batch
            batch_file = os.path.join(data_path, 'test_batch')
            self._load_batch(batch_file)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        # 所有数据加载完成后，堆叠成一个大的array
        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

        # 定义不同模式下的transform
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def _load_batch(self, batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']

            data = data.reshape(-1, 3, 32, 32)  # (N, C, H, W)
            data = np.transpose(data, (0, 2, 3, 1))  # (N, H, W, C)

            self.data.append(data)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = self.transform(img)
        return img, label


def get_loader(data_path, batch_size, mode='train', shuffle=True, num_workers=2):
    dataset = CIFAR10Dataset(data_path, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == 'train' else False,
        num_workers=num_workers
    )
    return dataloader


