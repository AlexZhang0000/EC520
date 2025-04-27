import numpy as np
import torch

def compute_iou(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)

    mask = labels != 255
    preds = preds[mask]
    labels = labels[mask]

    hist = torch.bincount(
        num_classes * labels + preds,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).float()

    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection

    iou = intersection / (union + 1e-6)
    miou = iou.mean().item()

    return iou.cpu().numpy(), miou
