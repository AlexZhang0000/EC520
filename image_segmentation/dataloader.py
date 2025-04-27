import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from PIL import Image
import io

# 自定义失真

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

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

# 自定义Dataset封装

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', distortion=None, download=True):
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)
        self.distortion = distortion

        # 图像变换
        self.input_transform_list = [
            T.Resize((512, 512)),  # ✅ 修改为512x512
            T.RandomHorizontalFlip(p=0.5),  # ✅ 加轻量flip
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ✅ 加颜色抖动
        ]

        if distortion:
            distortions = distortion.split('/')
            for d in distortions:
                if 'gaussianblur' in d:
                    params = d.split(':')[1]
                    kernel_size, sigma = params.split(',')
                    kernel_size = int(kernel_size)
                    sigma = float(sigma)
                    self.input_transform_list.append(T.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
                elif 'gaussiannoise' in d:
                    params = d.split(':')[1]
                    mean, std = params.split(',')
                    mean = float(mean)
                    std = float(std)
                    self.input_transform_list.append(AddGaussianNoise(mean=mean, std=std))
                elif 'aliasing' in d:
                    factor = int(d.split(':')[1])
                    self.input_transform_list.append(ApplyAliasing(factor=factor))
                elif 'jpegcompression' in d:
                    quality = int(d.split(':')[1])
                    self.input_transform_list.append(ApplyJPEGCompression(quality=quality))
                else:
                    raise ValueError(f"Unsupported distortion type: {d}")

        self.input_transform_list.append(T.ToTensor())
        self.input_transform = T.Compose(self.input_transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # 输入图像处理
        img = self.input_transform(img)

        # 标签也Resize到一样大小，注意不能乱变色
        target = target.resize((512, 512), Image.NEAREST)
        target = np.array(target)
        target = torch.from_numpy(target).long()

        return img, target

# Loader接口

def get_loader(root, batch_size=8, mode='train', shuffle=True, num_workers=2, distortion=None):
    dataset = VOCDataset(root=root, image_set=mode, distortion=distortion, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader

