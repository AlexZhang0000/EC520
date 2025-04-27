import os
import torch
from config import Config
from dataloader import get_loader
from model import UNet
from utils import compute_iou

def test():
    test_loader = get_loader(Config.data_root, batch_size=Config.batch_size, mode='test')

    model = UNet(n_classes=Config.num_classes).to(Config.device)
    model.load_state_dict(torch.load(os.path.join(Config.model_save_path, 'best_model.pth')))
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

    print(f"Test mIoU: {miou:.4f}")
    print(f"Per-class IoU: {per_class_iou}")

if __name__ == '__main__':
    test()
