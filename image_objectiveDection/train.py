import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed, CIoULoss, decode_boxes, box_iou

def map_target_to_feature_and_anchor(target, pred, feature_size, img_size):
    anchors_per_cell = 3
    pred = pred.view(anchors_per_cell, feature_size, feature_size, -1)

    x1, y1, x2, y2 = target
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    scale = feature_size / img_size
    grid_x = int(cx * scale)
    grid_y = int(cy * scale)

    grid_x = max(0, min(grid_x, feature_size - 1))
    grid_y = max(0, min(grid_y, feature_size - 1))

    gt_cx = cx
    gt_cy = cy
    gt_w = x2 - x1
    gt_h = y2 - y1
    gt_box = torch.tensor([gt_cx, gt_cy, gt_w, gt_h], device=pred.device).unsqueeze(0)

    max_iou = 0
    best_anchor = 0

    for anchor_idx in range(anchors_per_cell):
        pred_box = pred[anchor_idx, grid_y, grid_x, :4].unsqueeze(0)
        pred_box_decoded = decode_boxes(pred_box)
        gt_box_decoded = decode_boxes(gt_box)
        iou = box_iou(pred_box_decoded, gt_box_decoded)

        if iou.item() > max_iou:
            max_iou = iou.item()
            best_anchor = anchor_idx

    return grid_x, grid_y, best_anchor

def train(train_distortion=None):
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")

    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion, pin_memory=True)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)

    bce_loss = nn.BCEWithLogitsLoss()
    ciou_loss = CIoULoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )

    # 🚀 加上学习率调度器：Warmup + CosineAnnealing
    def lr_lambda(current_epoch):
        warmup_epochs = 5
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((current_epoch - warmup_epochs) / (Config.epochs - warmup_epochs) * 3.1415926535))).item()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_map = 0.0

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train_loader:
            imgs = imgs.to(Config.device)
            optimizer.zero_grad()
            preds = model(imgs)

            loss = 0.0
            batch_size = imgs.size(0)

            for b in range(batch_size):
                pred = preds[b]
                target_list = targets[b]
                label_list = labels[b]

                if len(label_list) == 0:
                    continue

                for obj_idx in range(len(label_list)):
                    target = target_list[obj_idx]
                    label = label_list[obj_idx]

                    if label < 0:
                        continue

                    feature_size = 40
                    img_size = Config.img_size

                    grid_x, grid_y, anchor_idx = map_target_to_feature_and_anchor(target, pred, feature_size, img_size)

                    pred = pred.view(3, feature_size, feature_size, -1)

                    pred_box = pred[anchor_idx, grid_y, grid_x, :4]
                    pred_obj = pred[anchor_idx, grid_y, grid_x, 4].unsqueeze(0)
                    pred_cls = pred[anchor_idx, grid_y, grid_x, 5:]

                    x1, y1, x2, y2 = target
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = (x2 - x1)
                    h = (y2 - y1)
                    true_box = torch.tensor([cx, cy, w, h], device=Config.device)

                    true_obj = torch.ones(1, device=Config.device)
                    true_cls = torch.nn.functional.one_hot(label, Config.num_classes).float().to(Config.device)

                    loc_loss = ciou_loss(pred_box.unsqueeze(0), true_box.unsqueeze(0))
                    obj_loss = bce_loss(pred_obj, true_obj)
                    cls_loss = bce_loss(pred_cls, true_cls)

                    # 🔥 Loss加权
                    loc_loss = 5.0 * loc_loss
                    obj_loss = 1.0 * obj_loss
                    cls_loss = 1.0 * cls_loss

                    loss += loc_loss + obj_loss + cls_loss

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
                imgs = imgs.to(Config.device)
                preds = model(imgs)
                for pred, target_list in zip(preds, targets):
                    preds_all.append(pred)
                    targets_all.append(target_list)

        mAP, precision, recall = compute_map(preds_all, targets_all)

        print(f"🧹 Epoch [{epoch}/{Config.epochs}] | Loss: {avg_loss:.4f} | Val mAP: {mAP:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

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







