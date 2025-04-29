# --- test_fast_improved.py ---

import os
import argparse
import torch
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

@torch.no_grad()
def test():
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")
    device = Config.device

    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    model = YOLOv5Backbone(num_classes=Config.num_classes).to(device)

    # 加载最优权重
    best_model_path = os.path.join(Config.model_save_path, "best_model_clean.pth")
    assert os.path.exists(best_model_path), f"Model not found: {best_model_path}"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds_all = []
    targets_all = []

    for imgs, targets, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)

        # 只用高分辨率 head 输出（20x20）
        preds = outputs[0]

        for pred, target_list in zip(preds, targets):
            preds_all.append(pred)
            targets_all.append(target_list)

    mAP, precision, recall = compute_map(preds_all, targets_all, iou_thresh=0.2)

    print("\n✅ Evaluation Results:")
    print(f"Val mAP: {mAP:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    test()



