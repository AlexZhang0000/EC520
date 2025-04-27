import os
import zipfile
import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image

def download_and_extract_camvid(root):
    google_drive_id = "1b2cg-8dMEd2S98RYJRx3MDCMXsKhSIEw"  # ✅CamVid压缩包ID
    zip_path = os.path.join(root, "camvid.zip")

    os.makedirs(root, exist_ok=True)

    # 下载
    print("Downloading CamVid dataset from Google Drive...")
    gdown.download(id=google_drive_id, output=zip_path, quiet=False)

    # 解压
    print("Extracting CamVid dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    # 删除zip
    os.remove(zip_path)
    print("CamVid download and extraction completed!")

class CamVidDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        if not os.path.exists(os.path.join(root, 'train')):  # 如果数据不存在
            download_and_extract_camvid(root)

        self.image_dir = os.path.join(root, mode)
        self.label_dir = os.path.join(root, mode + '_labels')
        self.image_names = sorted(os.listdir(self.image_dir))

        self.input_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        img = self.input_transform(img)

        label = label.resize((256, 256), Image.NEAREST)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return img, label

def get_loader(root, batch_size=16, mode='train', shuffle=True, num_workers=2):
    dataset = CamVidDataset(root=root, mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader



