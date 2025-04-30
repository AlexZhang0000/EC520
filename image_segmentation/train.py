import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou

def train(train_distortion=None):
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    # GPU检测和动态batch size
    num_gpus = torch.cuda.device_count()
    print(f"✅ Detected {num_gpus} GPU(s).")

    base_batch_size = 16
    effective_batch_size = base_batch_size * max(1, num_gpus)
    print(f"✅ Using batch size: {effective_batch_size}")

    # Data loaders
    train_loader = get_loader(Config.data_root, batch_size=effective_batch_size, mode='train', distortion=train_distortion)
    val_loader = get_loader(Config.data_root, batch_size=effective_batch_size, mode='val', distortion=None)

    # 模型 + 多GPU并行
    model = UNet(n_classes=Config.num_classes)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(Config.device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_miou = 0.0

    # 日志路径
    distortion_tag = 'clean' if train_distortion is None else train_distortion.replace(':', '_').replace('/', '_')
    log_filename = f"train_log_{distortion_tag}.csv"
    log_filepath = os.path.join(Config.result_save_path, log_filename)
    os.makedirs(Config.result_save_path, exist_ok=True)

    # 写入CSV表头
    with open(log_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_miou'])

    for epoch in range(1, Config.num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(Config.device)
            masks = masks.to(Config.device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        preds_all = []
        masks_all = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.device)
                masks = masks.to(Config.device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                preds_all.append(preds.cpu())
                masks_all.append(masks.cpu())

        preds_all = torch.cat(preds_all, dim=0)
        masks_all = torch.cat(masks_all, dim=0)

        _, val_miou = compute_iou(preds_all, masks_all, num_classes=Config.num_classes)

        print(f"Epoch [{epoch}/{Config.num_epochs}] | Train Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # 记录 val_mIoU 到日志文件
        with open(log_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_miou])

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            os.makedirs(Config.model_save_path, exist_ok=True)

            if train_distortion is None:
                model_name = 'best_model.pth'
            else:
                distortion_name = train_distortion.replace(':', '_').replace('/', '_')
                model_name = f'best_model_{distortion_name}.pth'

            torch.save(model.state_dict(), os.path.join(Config.model_save_path, model_name))

        scheduler.step()

    print('Training finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None,
                        help="Specify distortion for training (optional), e.g., gaussianblur:5,1.0 / aliasing:4 / jpegcompression:20")
    args = parser.parse_args()

    train(train_distortion=args.train_distortion)






