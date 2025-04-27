import os
import argparse
import torch
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

def test(model_suffix='', test_distortion=None):
    set_seed(Config.seed)

    print(f"✅ Using device: {Config.device}")

    model_filename = f"best_model{model_suffix}.pth"
    model_path = os.path.join(Config.model_save_path, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=test_distortion, pin_memory=True)

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
            for pred, target_list in zip(preds, targets):
                preds_all.append(pred)
                targets_all.append(target_list)

    mAP, precision, recall = compute_map(preds_all, targets_all)

    print(f"🏁 Test Results -> mAP: {mAP:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_suffix', type=str, default='', help='Suffix for the model filename')
    parser.add_argument('--test_distortion', type=str, default=None, help='Test-time distortion type')
    args = parser.parse_args()

    test(model_suffix=args.model_suffix, test_distortion=args.test_distortion)



