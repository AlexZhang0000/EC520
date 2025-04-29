

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

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    best_map = 0.0

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # [high-res, low-res]

            loss = None
            batch_size = imgs.size(0)

            for b in range(batch_size):
                target_list = targets[b]
                label_list = labels[b]

                if len(target_list) == 0:
                    continue

                for obj_idx in torch.randperm(len(label_list))[:min(10, len(label_list))]:
                    target = target_list[obj_idx]
                    label = label_list[obj_idx]

                    label = int(label)
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

                    pred = best_out.permute(1, 2, 0).contiguous().view(best_out.shape[1], best_out.shape[2], 3, -1)
                    pred = pred.permute(2, 0, 1, 3).contiguous()

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

                    true_box = torch.tensor([cx_gt, cy_gt, w_gt, h_gt], device=device)
                    true_obj = torch.ones(1, device=device)
                    label = torch.tensor(label, device=device)
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

    print("✅ Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_distortion', type=str, default=None, help='Train distortion type (optional)')
    args = parser.parse_args()

    train(train_distortion=args.train_distortion)










