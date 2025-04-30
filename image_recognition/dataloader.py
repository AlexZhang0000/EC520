import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import io

# --- 自定义失真方法 ---

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class ApplyAliasing(torch.nn.Module):
    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        w, h = img.size
        new_w, new_h = w // self.factor, h // self.factor
        img = img.resize((new_w, new_h), resample=Image.NEAREST)
        img = img.resize((w, h), resample=Image.NEAREST)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(factor={self.factor})"

class ApplyJPEGCompression(torch.nn.Module):
    def __init__(self, quality=75):
        super().__init__()
        self.quality = quality

    def forward(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality})"

# --- 主Dataset定义 ---

class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, mode='train', distortion=None):
        self.data = []
        self.labels = []
        self.mode = mode
        self.distortion = distortion

        # 加载数据
        if mode == 'train':
            for i in range(1, 5):
                batch_file = os.path.join(data_path, f'data_batch_{i}')
                self._load_batch(batch_file)
        elif mode == 'val':
            batch_file = os.path.join(data_path, 'validation_batch')
            self._load_batch(batch_file)
        elif mode == 'test':
            batch_file = os.path.join(data_path, 'test_batch')
            self._load_batch(batch_file)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

        # 定义 transform
        transform_list = []
        tensor_transforms = []

        if self.mode == 'train':
            transform_list += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]

        if self.distortion:
            distortions = self.distortion.split('/')
            for d in distortions:
                if 'gaussianblur' in d:
                    params = d.split(':')[1]
                    kernel_size, sigma = params.split(',')
                    kernel_size = int(kernel_size)
                    sigma = float(sigma)
                    transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
                elif 'aliasing' in d:
                    factor = int(d.split(':')[1])
                    transform_list.append(ApplyAliasing(factor=factor))
                elif 'jpegcompression' in d:
                    quality = int(d.split(':')[1])
                    transform_list.append(ApplyJPEGCompression(quality=quality))
                elif 'gaussiannoise' in d:
                    params = d.split(':')[1]
                    mean, std = params.split(',')
                    mean = float(mean)
                    std = float(std)
                    tensor_transforms.append(AddGaussianNoise(mean=mean, std=std))
                else:
                    raise ValueError(f"Unsupported distortion type: {d}")

        # 顺序：PIL变换 → ToTensor → Tensor变换 → Normalize
        transform_list += [
            transforms.ToTensor()
        ]
        transform_list += tensor_transforms
        transform_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.transform = transforms.Compose(transform_list)

    def _load_batch(self, batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']

            data = data.reshape(-1, 3, 32, 32)
            data = np.transpose(data, (0, 2, 3, 1))

            self.data.append(data)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

# --- DataLoader封装 ---

def get_loader(data_path, batch_size, mode='train', shuffle=True, num_workers=2, distortion=None):
    dataset = CIFAR10Dataset(data_path, mode, distortion)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == 'train' else False,
        num_workers=num_workers
    )
    return dataloader






