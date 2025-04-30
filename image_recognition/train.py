import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dataloader import get_loader
from model import ResNet18
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None,
                        help="Distortion format for training data: gaussianblur:kernel_size,sigma/...")
    return parser.parse_args()

def make_suffix(distortion, mode='train'):
    if distortion is None:
        return '_clean'
    clean_distortion = distortion.replace('/', '_').replace(':', '_').replace(',', '_')
    return f"_distorted_{clean_distortion}"

def train():
    args = parse_args()

    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    os.makedirs(Config.model_save_path, exist_ok=True)
    os.makedirs(Config.results_save_path, exist_ok=True)

    # === 自动检测 GPU 数量并更新 device 和 batch_size ===
    num_gpus = torch.cuda.device_count()
    print(f"✅ Detected {num_gpus} GPU(s)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 动态 batch size
    batch_size = Config.batch_size * num_gpus if num_gpus > 0 else Config.batch_size

    # Prepare data loaders
    train_loader = get_loader(Config.train_data_dir, batch_size=batch_size, mode='train', distortion=args.train_distortion)
    val_loader = get_loader(Config.test_data_dir, batch_size=batch_size, mode='val', distortion=None)  # validation always clean!

    model = ResNet18(num_classes=Config.num_classes)

    # 多GPU支持
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate,
                          momentum=Config.momentum, weight_decay=Config.weight_decay)

    best_val_acc = 0.0
    save_suffix = make_suffix(args.train_distortion)

    for epoch in range(1, Config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        print(f'Epoch [{epoch}/{Config.num_epochs}] '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% '
              f'| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_name = f"best_model{save_suffix}.pth"
            torch.save(model.state_dict(), os.path.join(Config.model_save_path, save_name))

    print('✅ Training finished.')

if __name__ == '__main__':
    train()



