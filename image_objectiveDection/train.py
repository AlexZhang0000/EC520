import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

def map_target_to_feature(target, feature_size, img_size):
    """
    Map ground truth (x1, y1, x2, y2) to feature map (grid_x, grid_y) index.
    防止grid越界。
    """
    x1, y1, x2, y2 = target  # 输入是 (x1, y1, x2, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    scale = feature_size / img_size
    grid_x = int(cx * scale)
    grid_y = int(cy * scale)

    # 防止grid越界（比如x=img_size边界）
    grid_x = max(0, min(grid_x, feature_size - 1))
    grid_y = max(0, min(grid_y, feature_size - 1))

    return grid_x, grid_y

def train(train_distortion=None):
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")

    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion, pin_memory=True)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )

    best_map = 0.0

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(Config.device)
            optimizer.zero_grad()
            preds = model(imgs)  # preds shape: (batch, 1200, 5+num_classes)

            loss = 0.0

            batch_size = imgs.size(0)

            for b in range(batch_size):
                pred = preds[b]  # (1200, 5+num_classes)
                target = targets[b]
                label = labels[b]

                if label[0] == -1:
                    continue

                feature_size = 20  # 你的输出feature map大小是20x20
                img_size = Config.img_size  # 输入图像大小640

                grid_x, grid_y = map_target_to_feature(target, feature_size, img_size)

                anchor_idx = 0  # 简化版：只用第0个anchor

                pred_idx = (anchor_idx * feature_size * feature_size) + (grid_y * feature_size) + grid_x

                pred_box = pred[pred_idx, :4]  # (cx, cy, w, h)
                pred_obj = pred[pred_idx, 4]
                pred_cls = pred[pred_idx, 5:]

                # ground truth
                x1, y1, x2, y2 = target
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                true_box = torch.tensor([cx, cy, w, h], device=Config.device)

                true_obj = torch.ones(1, device=Config.device)
                true_cls = torch.nn.functional.one_hot(label, Config.num_classes).float().to(Config.device)

                loc_loss = mse_loss(pred_box, true_box)
                obj_loss = bce_loss(pred_obj, true_obj)
                cls_loss = bce_loss(pred_cls, true_cls)

                loss += loc_loss + obj_loss + cls_loss

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




