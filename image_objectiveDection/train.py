# ✅ 加速版 train.py
# 包含优化：样本限制、提前 reshape、AMP 混合精度、Dataloader 设置优化

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if num_gpus > 0 else "cpu")
    print(f"✅ Found {num_gpus} GPU(s). Using device: {device}")

    effective_batch_size = Config.batch_size * max(1, num_gpus)
    print(f"✅ Effective Batch Size: {effective_batch_size}")

    train_loader = get_loader(batch_size=effective_batch_size, mode='train', distortion=train_distortion, pin_memory=True, num_workers=4)
    val_loader = get_loader(batch_size=effective_batch_size, mode='val', distortion=None, pin_memory=True, num_workers=4)

    model = YOLOv5Backbone(num_classes=Config.num_classes)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    box_loss = CIoULoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    scaler = torch.amp.GradScaler(device_type='cuda') if torch.cuda.is_available() else None


    best_map = 0.0
    log_suffix = train_distortion.replace('/', '_').replace(':', '_').replace(',', '_') if train_distortion else 'clean'
    log_path = os.path.join(Config.result_save_dir, f"train_log_{log_suffix}.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,val_map\n")

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(imgs)

            loss = 0.0
            batch_size = imgs.size(0)

            for b in range(min(batch_size, 4)):
                if len(labels[b]) == 0:
                    continue

                target = targets[b][0]
                label = int(labels[b][0])
                if label >= Config.num_classes:
                    continue

                cx = (target[0] + target[2]) / 2
                cy = (target[1] + target[3]) / 2

                best_out = None
                best_grid_x = None
                best_grid_y = None

                for pred in outputs:
                    _, h, w, _ = pred.shape
                    grid_x = int(cx * w)
                    grid_y = int(cy * h)
                    if 0 <= grid_x < w and 0 <= grid_y < h:
                        best_out = pred[b]
                        best_grid_x = grid_x
                        best_grid_y = grid_y
                        break

                if best_out is None:
                    continue

                C = best_out.shape[0]  # 通道数
                _, h, w = best_out.shape
                pred = best_out.view(C, h, w).permute(1, 2, 0).contiguous()  # (h, w, C)
                num_anchors = 3
                num_outputs = C // num_anchors
                pred = pred.view(h, w, num_anchors, num_outputs).permute(2, 0, 1, 3).contiguous()  # (3, h, w, num_outputs)

                pred_box = pred[0, best_grid_y, best_grid_x, :4]
                pred_obj = pred[0, best_grid_y, best_grid_x, 4].unsqueeze(0)
                pred_cls = pred[0, best_grid_y, best_grid_x, 5:]

                if torch.any(torch.isnan(pred_box)) or torch.any(torch.isinf(pred_box)):
                    continue

                img_size = Config.img_size
                cx_gt = (target[0] + target[2]) / 2 * img_size
                cy_gt = (target[1] + target[3]) / 2 * img_size
                w_gt = (target[2] - target[0]) * img_size
                h_gt = (target[3] - target[1]) * img_size
                if w_gt <= 1 or h_gt <= 1:
                    continue

                true_box = torch.tensor([cx_gt, cy_gt, w_gt, h_gt], device=device)
                true_obj = torch.ones(1, device=device)
                true_cls = torch.nn.functional.one_hot(torch.tensor(label, device=device), Config.num_classes).float()

                loc_loss = 2.0 * box_loss(pred_box.unsqueeze(0), true_box.unsqueeze(0))
                obj_loss = 1.0 * bce_loss(pred_obj, true_obj)
                cls_loss = 1.0 * bce_loss(pred_cls, true_cls)
                loss += (loc_loss + obj_loss + cls_loss)

            if loss > 0:
                loss = loss / batch_size
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                outputs = model(imgs)
                preds = outputs[0]
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

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{mAP:.8f}\n")

    print("✅ Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None, help='Train distortion type (optional)')
    args = parser.parse_args()
    train(train_distortion=args.train_distortion)










