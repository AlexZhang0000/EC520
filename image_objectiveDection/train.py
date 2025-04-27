import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

def train(train_distortion=None):
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")

    # 加载数据
    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion, pin_memory=True)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("❌ Error: train_loader or val_loader is empty.")

    # 初始化模型
    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)

    # 定义损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    ciou_loss = nn.MSELoss()  # 注意：用MSE代替真实CIoU loss，正式版可以升级

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )

    best_map = 0.0  # 保存最佳mAP

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(Config.device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = 0.0
            for i in range(len(imgs)):
                pred = preds[i]
                target = targets[i]
                label = labels[i]

                if label[0] == -1:
                    continue  # 如果没有合法目标，跳过

                # 定位损失
                box_pred = pred[..., :4]
                box_true = target.to(Config.device)
                loc_loss = ciou_loss(box_pred, box_true)

                # 置信度损失
                obj_loss = bce_loss(pred[..., 4], torch.ones_like(pred[..., 4]))

                # 不做分类loss
                loss += loc_loss + obj_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for imgs, targets, labels in val_loader:
                imgs = imgs.to(Config.device)
                preds = model(imgs)
                preds_all.extend(preds)
                targets_all.extend(targets)

        mAP, precision, recall = compute_map(preds_all, targets_all)

        print(f"🧹 Epoch [{epoch}/{Config.epochs}] | Loss: {avg_loss:.4f} | Val mAP: {mAP:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # 保存最佳模型
        if mAP > best_map:
            best_map = mAP
            save_filename = f"best_model{'_' + train_distortion if train_distortion else '_clean'}.pth"
            save_path = os.path.join(Config.base_save_dir, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")

    print("✅ Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None, help='Train distortion type (optional)')
    args = parser.parse_args()

    train(train_distortion=args.train_distortion)


