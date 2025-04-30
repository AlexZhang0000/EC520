import os
import argparse
import torch
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

@torch.no_grad()
def test(model_suffix="_clean", test_distortion=None):
    set_seed(Config.seed)

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if num_gpus > 0 else "cpu")
    print(f"✅ Found {num_gpus} GPU(s). Using device: {device}")

    val_loader = get_loader(batch_size=Config.batch_size * max(1, num_gpus), mode='val',
                            distortion=test_distortion, pin_memory=True, num_workers=4)

    model = YOLOv5Backbone(num_classes=Config.num_classes)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 加载模型权重
    model_path = os.path.join(Config.model_save_path, f"best_model{model_suffix}.pth")
    assert os.path.exists(model_path), f"❌ Model not found: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_all = []
    targets_all = []

    print("🚀 Starting Evaluation...")
    for imgs, targets, labels in val_loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(imgs)
        preds = outputs[0]
        for pred, target_list in zip(preds, targets):
            preds_all.append(pred)
            targets_all.append(target_list)

    mAP, precision, recall = compute_map(preds_all, targets_all, iou_thresh=0.2)

    print("\n✅ Evaluation Results:")
    print(f"Val mAP: {mAP:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall: {recall:.4f}")

    # ✅ 保存结果
    distortion_tag = test_distortion.replace(':', '_').replace(',', '_') if test_distortion else None
    log_name = f"test_log{model_suffix}{'_' + distortion_tag if distortion_tag else ''}.txt"
    log_path = os.path.join(Config.result_save_dir, log_name)
    with open(log_path, 'w') as f:
        f.write(f"Model: {model_suffix}\n")
        f.write(f"Val mAP: {mAP:.8f}\n")
        f.write(f"Precision: {precision:.8f}\n")
        f.write(f"Recall: {recall:.4f}\n")
    print(f"📁 Test log saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_suffix', type=str, default="_clean", help='Suffix of model file to load')
    parser.add_argument('--test_distortion', type=str, default=None, help='Distortion type applied in validation')
    args = parser.parse_args()

    test(model_suffix=args.model_suffix, test_distortion=args.test_distortion)


