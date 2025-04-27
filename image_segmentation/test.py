import os
import torch
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou, print_iou
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_distortion', type=str, default=None,
                        help="Distortion format for test data: gaussianblur:5,2.0 / gaussiannoise:0,0.1 / aliasing:4 / jpegcompression:20")
    parser.add_argument('--model_suffix', type=str, default='_clean',
                        help="Suffix for model file to load, e.g., _clean, _distorted_gaussianblur_5_2.0")
    return parser.parse_args()

def make_suffix(distortion):
    if distortion is None:
        return ''
    clean_distortion = distortion.replace('/', '_').replace(':', '_').replace(',', '_')
    return f"_test_{clean_distortion}"

def test():
    args = parse_args()
    distortion_suffix = make_suffix(args.test_distortion)

    test_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='val', distortion=args.test_distortion)

    model = UNet(n_classes=Config.num_classes).to(Config.device)
    model_path = os.path.join(Config.model_save_path, f'best_model{args.model_suffix}.pth')
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

    ious, miou = compute_iou(preds_all, masks_all, num_classes=Config.num_classes)

    print(f"\nFinal mean IoU (mIoU): {miou:.4f}")
    print_iou(ious)

    # 保存测试结果
    os.makedirs(Config.results_save_path, exist_ok=True)
    save_path = os.path.join(Config.results_save_path, f'test_results{distortion_suffix}.txt')
    with open(save_path, 'w') as f:
        f.write(f"Final mean IoU (mIoU): {miou:.4f}\n")
        f.write("Per-Class IoU:\n")
        for idx, iou in enumerate(ious):
            f.write(f"Class {idx}: {iou:.4f}\n")

if __name__ == '__main__':
    test()
