import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map

def train(train_distortion=None):
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    # 数据
    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None)

    # 模型
    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)

    # 损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    ciou_loss = nn.MSELoss()  # 简化版，正式应该用CIoU，先用MSE代替示范

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=Config.momentum, weight_decay=Config.weight_decay)

    # 保存最好mAP
    best_map = 0.0

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

                if label[0] == -1:  # 如果没有合法目标
                    continue

                # 简单示范版 loss：
                # 1. 分类BCE
                cls_loss = bce_loss(pred[..., 5:], torch.nn.functional.one_hot(label, Config.num_classes).float().to(Config.device))
                # 2. 定位 (用MSE临时代替IoU loss)
                box_pred = pred[..., :4]
                box_true = target.to(Config.device)
                loc_loss = ciou_loss(box_pred, box_true)
                # 3. 置信度
                obj_loss = bce_loss(pred[..., 4], torch.ones_like(pred[..., 4]))

                loss += cls_loss + loc_loss + obj_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证集评估
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

        print(f"Epoch [{epoch}/{Config.epochs}] | Loss: {avg_loss:.4f} | Val mAP: {mAP:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # 保存
        if mAP > best_map:
            best_map = mAP
            save_path = os.path.join(Config.model_save_path, f"best_model{'_' + train_distortion if train_distortion else ''}.pth")
            torch.save(model.state_dict(), save_path)

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None, help='train distortion type (optional)')
    args = parser.parse_args()

    train(train_distortion=args.train_distortion)
