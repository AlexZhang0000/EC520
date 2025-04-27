import os
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation
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

# --- 选定的Pascal VOC子集类别 ---

VOC_TARGET_CLASSES = {
    15: 1,  # person
    8: 2,   # cat
    12: 3,  # dog
    7: 4    # car
}
# 其他类别全部设为 ignore_index=255

# --- 主Dataset类 ---

class VOCSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', download=True, distortion=None):
        self.dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)
        self.distortion = distortion

        self.input_transform_list = [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip()
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
        img, mask = self.dataset[idx]

        img = self.input_transform(img)

        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask)

        # 重映射
        remapped_mask = np.full_like(mask, fill_value=255)
        for voc_class, mapped_class in VOC_TARGET_CLASSES.items():
            remapped_mask[mask == voc_class] = mapped_class
        remapped_mask[mask == 0] = 0

        remapped_mask = torch.from_numpy(remapped_mask).long()

        return img, remapped_mask

# --- Loader接口 ---

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2, distortion=None):
    dataset = VOCSubsetDataset(root=root, image_set='train' if mode == 'train' else 'val', download=True, distortion=distortion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
    return loader






