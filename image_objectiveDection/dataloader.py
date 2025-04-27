import os
import numpy as np
import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import io

# --- 自定义失真类 ---

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

class ApplyAliasing(torch.nn.Module):
    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        w, h = img.size
        img = img.resize((w // self.factor, h // self.factor), resample=Image.NEAREST)
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

# --- 目标类别映射 ---

VOC_TARGET_CLASSES = {
    'person': 0,
    'cat': 1,
    'dog': 2,
    'car': 3,
    'bicycle': 4
}
# 最终类别数 = 5类

# --- 主Dataset ---

class VOCDetectionSubset(torch.utils.data.Dataset):
    def __init__(self, root='./data', year='2007', image_set='train', download=True, distortion=None, img_size=640):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.distortion = distortion
        self.img_size = img_size

        self.input_transform_list = [
            T.Resize((img_size, img_size)),
        ]

        if self.distortion:
            distortions = self.distortion.split('/')
            for d in distortions:
                if 'gaussianblur' in d:
                    params = d.split(':')[1]
                    kernel_size, sigma = params.split(',')
                    kernel_size = int(kernel_size)
                    sigma = float(sigma)
                    self.input_transform_list.append(T.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
                elif 'gaussiannoise' in d:
                    mean, std = map(float, d.split(':')[1].split(','))
                    self.input_transform_list.append(AddGaussianNoise(mean=mean, std=std))
                elif 'aliasing' in d:
                    factor = int(d.split(':')[1])
                    self.input_transform_list.append(ApplyAliasing(factor=factor))
                elif 'jpegcompression' in d:
                    quality = int(d.split(':')[1])
                    self.input_transform_list.append(ApplyJPEGCompression(quality=quality))
                else:
                    raise ValueError(f"Unsupported distortion type: {d}")

        self.input_transform_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]

        self.input_transform = T.Compose(self.input_transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # 筛选目标
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        boxes = []
        labels = []

        for obj in objs:
            cls_name = obj['name'].lower().strip()
            if cls_name not in VOC_TARGET_CLASSES:
                continue

            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_TARGET_CLASSES[cls_name])

        if len(boxes) == 0:
            # 没有符合的目标，跳过，随便给一个框防止出错
            boxes = torch.zeros((1, 4))
            labels = torch.full((1,), -1)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img = self.input_transform(img)

        # 坐标缩放到 [0,1]
        _, h, w = img.shape
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        boxes = torch.clamp(boxes, 0., 1.)

        return img, boxes, labels

# --- Loader封装 ---

def get_loader(root='./data', batch_size=16, mode='train', shuffle=True, num_workers=2, distortion=None):
    dataset = VOCDetectionSubset(root=root, year='2007', image_set=mode, download=True, distortion=distortion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers, collate_fn=collate_fn)
    return loader

def collate_fn(batch):
    imgs, targets, labels = list(zip(*batch))
    return torch.stack(imgs, dim=0), targets, labels
