import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None,
                        help="Distortion format for training data: gaussianblur:5,2.0 / gaussiannoise:0,0.1 / aliasing:4 / jpegcompression:20")
    return parser.parse_args()

def make_suffix(distortion):
    if distortion is None:
        return '_clean'
    clean_distortion = distortion.replace('/', '_').replace(':', '_').replace(',', '_')
    return f"_distorted_{clean_distortion}"

def train():
    args = parse_args()
    save_suffix = make_suffix(args.train_distortion)

    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    # Data loaders
    train_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='train', distortion=args.train_distortion)
    val_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='val', distortion=None)  # val是干净的！

    model = UNet(n_classes=Config.num_classes).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    best_miou = 0.0

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

        # Validation
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

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_name = f"best_model{save_suffix}.pth"
            torch.save(model.state_dict(), os.path.join(Config.model_save_path, save_name))

    print('Training finished.')

if __name__ == '__main__':
    train()
