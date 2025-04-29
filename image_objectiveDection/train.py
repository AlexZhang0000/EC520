# --- train_fast_improved.py ---

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed, decode_boxes, box_iou

class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, true_boxes):
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        true_x1 = true_boxes[:, 0] - true_boxes[:, 2] / 2
        true_y1 = true_boxes[:, 1] - true_boxes[:, 3] / 2
        true_x2 = true_boxes[:, 0] + true_boxes[:, 2] / 2
        true_y2 = true_boxes[:, 1] + true_boxes[:, 3] / 2

        inter_x1 = torch.max(pred_x1, true_x1)
        inter_y1 = torch.max(pred_y1, true_y1)
        inter_x2 = torch.min(pred_x2, true_x2)
        inter_y2 = torch.min(pred_y2, true_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)

        union_area = pred_area + true_area - inter_area
        iou = inter_area / (union_area + 1e-6)

        center_dist = (pred_boxes[:, 0] - true_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - true_boxes[:, 1]) ** 2
        enclose_x1 = torch.min(pred_x1, true_x1)
        enclose_y1 = torch.min(pred_y1, true_y1)
        enclose_x2 = torch.max(pred_x2, true_x2)
        enclose_y2 = torch.max(pred_y2, true_y2)
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

        ciou = iou - center_dist / (enclose_diagonal + 1e-6)
        loss = 1 - ciou
        return loss.mean()

def train(train_distortion=None):
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")
    device = Config.device

    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion, pin_memory=True)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    model = YOLOv5Backbone(num_classes=Config.num_classes).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    box_loss = CIoULoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    best_map = 0.0

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            preds = model(imgs)

            loss = None
            batch_size = imgs.size(0)

            for b in range(batch_size):
                pred = preds[b]
                target_list = targets[b]
                label_list = labels[b]

                if len(target_list) == 0:
                    continue

                num_objs = min(10, len(label_list))
                for obj_idx in torch.randperm(len(label_list))[:num_objs]:
                    target = target_list[obj_idx]
                    label = label_list[obj_idx]

                    feature_size = 40
                    img_size = Config.img_size

                    try:
                        grid_x = int(((target[0] + target[2]) / 2) * feature_size)
                        grid_y = int(((target[1] + target[3]) / 2) * feature_size)
                        anchor_idx = 0
                    except Exception as e:
                        continue

                    pred = pred.view(3, feature_size, feature_size, -1)

                    pred_box = pred[anchor_idx, grid_y, grid_x, :4]
                    pred_box = pred_box.clamp(0, img_size)

                    pred_obj = pred[anchor_idx, grid_y, grid_x, 4].unsqueeze(0)
                    pred_cls = pred[anchor_idx, grid_y, grid_x, 5:]

                    x1, y1, x2, y2 = target
                    cx = (x1 + x2) / 2 * img_size
                    cy = (y1 + y2) / 2 * img_size
                    w = (x2 - x1) * img_size
                    h = (y2 - y1) * img_size
                    true_box = torch.tensor([cx, cy, w, h], device=device)

                    true_obj = torch.ones(1, device=device)
                    label = torch.tensor(int(label), device=device)
                    true_cls = torch.nn.functional.one_hot(label, Config.num_classes).float()

                    loc_loss = 2.0 * box_loss(pred_box.unsqueeze(0), true_box.unsqueeze(0))
                    obj_loss = 1.0 * bce_loss(pred_obj, true_obj)
                    cls_loss = 1.0 * bce_loss(pred_cls, true_cls)

                    if loss is None:
                        loss = loc_loss + obj_loss + cls_loss
                    else:
                        loss = loss + (loc_loss + obj_loss + cls_loss)

            if loss is not None:
                loss = loss / batch_size
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for imgs, targets, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                for pred, target_list in zip(preds, targets):
                    preds_all.append(pred)
                    targets_all.append(target_list)

        mAP, precision, recall = compute_map(preds_all, targets_all, iou_thresh=0.2)

        print(f"🧹 Epoch [{epoch}/{Config.epochs}] | Loss: {avg_loss:.6f} | Val mAP: {mAP:.8f} | Precision: {precision:.8f} | Recall: {recall:.4f}")

        if mAP > best_map:
            best_map = mAP
            save_filename = f"best_model{'_' + train_distortion if train_distortion else '_clean'}.pth"
            save_path = os.path.join(Config.model_save_path, save_filename)
            torch.save(model.state_dict(), save_path)

    print("✅ Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None, help='Train distortion type (optional)')
    args = parser.parse_args()

    train(train_distortion=args.train_distortion)









