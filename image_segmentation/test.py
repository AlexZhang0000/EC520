import argparse
import os
import torch
import numpy as np
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou

def test(model_path, distortion=None):
    # === 加载模型 ===
    print(f"🔍 Loading model: {model_path}")
    model = UNet(n_classes=Config.num_classes).to(Config.device)

    # === 加载模型参数（兼容 DataParallel）===
    state_dict = torch.load(model_path, map_location=Config.device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.eval()

    # === 解析训练集标签（根据模型文件名）===
    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    model_tag = model_tag.replace("best_model", "").lstrip("_") or "clean"
    print(f"🔍 Loading model trained on: {model_tag}")

    # === 打印测试集失真标签 ===
    dist_tag = distortion.replace(":", "_").replace(",", "_") if distortion else "clean"
    print(f"🧪 Testing with distortion: {distortion or 'clean'}")

    # === 加载验证集 ===
    val_loader = get_loader(Config.data_root, batch_size=1, mode='val', distortion=distortion)

    preds_all = []
    masks_all = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(Config.device)
            masks = masks.to(Config.device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            preds_all.append(preds.cpu())
            masks_all.append(masks.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    masks_all = torch.cat(masks_all, dim=0)

    # === 计算 mIoU ===
    per_class_iou, miou = compute_iou(preds_all, masks_all, num_classes=Config.num_classes)
    print(f"✅ mIoU: {miou:.4f}")

    # === 保存结果 ===
    os.makedirs(Config.result_save_path, exist_ok=True)
    result_path = os.path.join(Config.result_save_path, f"result_{model_tag}_{dist_tag}.txt")
    with open(result_path, 'w') as f:
        f.write(f"mIoU: {miou:.4f}\n")
        for i, iou in enumerate(per_class_iou):
            f.write(f"Class {i}: IoU = {iou:.4f}\n")

    print(f"📄 Saved result to: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to saved model (.pth)")
    parser.add_argument("--distortion", type=str, default=None, help="Optional test distortion (e.g. blur:2)")
    args = parser.parse_args()

    test(model_path=args.model_path, distortion=args.distortion)


