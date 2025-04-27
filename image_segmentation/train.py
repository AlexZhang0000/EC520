import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou
from tqdm import tqdm

def train():
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    train_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='train')
    val_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='test')

    model = UNet(n_classes=Config.num_classes).to(Config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_miou = 0.0

    for epoch in range(1, Config.num_epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{Config.num_epochs}]', leave=False)

        for images, masks in loop:
            images = images.to(Config.device)
            masks = masks.to(Config.device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

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

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(Config.model_save_path, 'best_model.pth'))

        scheduler.step()

    print('Training finished.')

if __name__ == '__main__':
    train()





