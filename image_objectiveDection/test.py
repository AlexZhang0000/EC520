import os
import argparse
import torch
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map

def test(model_path, distortion=None):
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    # 加载验证集（可以加distortion）
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=distortion)

    # 加载模型
    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)
    checkpoint = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(checkpoint)
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for imgs, targets, labels in val_loader:
            imgs = imgs.to(Config.device)
            preds = model(imgs)
            preds_all.extend(preds)
            targets_all.extend(targets)

    # 计算mAP, precision, recall
    mAP, precision, recall = compute_map(preds_all, targets_all)

    print(f"Test Results -> mAP: {mAP:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--distortion', type=str, default=None, help='distortion type during testing (optional)')
    args = parser.parse_args()

    test(model_path=args.model_path, distortion=args.distortion)
