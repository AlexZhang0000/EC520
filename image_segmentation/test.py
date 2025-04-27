import os
import torch
import argparse
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou

def test(model_path, distortion=None):
    test_loader = get_loader(Config.data_root, batch_size=1, mode='val', shuffle=False, distortion=distortion)

    model = UNet(n_classes=Config.num_classes).to(Config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds_all = []
    masks_all = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(Config.device)
            masks = masks.to(Config.device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            preds_all.append(preds.cpu())
            masks_all.append(masks.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    masks_all = torch.cat(masks_all, dim=0)

    per_class_iou, miou = compute_iou(preds_all, masks_all, num_classes=Config.num_classes)

    os.makedirs(Config.result_save_path, exist_ok=True)

    # 保存结果名包含模型文件名
    model_name = os.path.basename(model_path).replace('.pth', '')
    distortion_name = 'clean' if distortion is None else distortion.replace(':', '_').replace('/', '_')
    save_name = f"result_{model_name}_{distortion_name}.txt"
    save_path = os.path.join(Config.result_save_path, save_name)

    with open(save_path, 'w') as f:
        f.write(f"mIoU: {miou:.4f}\n")
        for idx, iou in enumerate(per_class_iou):
            f.write(f"Class {idx}: IoU = {iou:.4f}\n")

    print(f"Test Finished. mIoU = {miou:.4f}")
    print(f"Results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to model checkpoint (e.g., saved_models/best_model.pth)")
    parser.add_argument('--distortion', type=str, default=None,
                        help="Specify distortion, e.g., gaussianblur:5,1.0 / gaussiannoise:0,0.1 / aliasing:4 / jpegcompression:50")
    args = parser.parse_args()

    test(model_path=args.model_path, distortion=args.distortion)

