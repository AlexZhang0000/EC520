import torch
import random
import numpy as np

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: (N, 4)
    box2: (M, 4)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou  # (N, M)

def compute_map(preds_all, targets_all, iou_thresh=0.5):
    tp = 0
    fp = 0
    fn = 0

    for preds, targets in zip(preds_all, targets_all):
        preds = preds.sigmoid()

        obj_conf = preds[..., 4]
        class_conf = preds[..., 5:]

        pred_mask = obj_conf > 0.5

        if pred_mask.sum() == 0:
            if isinstance(targets, torch.Tensor):
                if targets.ndim == 1 and targets.size(0) == 4:
                    fn += 1
                else:
                    fn += len(targets)
            else:
                fn += len(targets)
            continue

        pred_boxes = preds[pred_mask][..., :4]
        pred_boxes = decode_boxes(pred_boxes)

        # 🔥 保证 true_boxes 一定是 (N, 4) 格式
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1 and targets.size(0) == 4:
                true_boxes = targets.unsqueeze(0).to(pred_boxes.device)  # (1,4)
            else:
                true_boxes = targets.to(pred_boxes.device)  # 已经是(N,4)
        else:
            true_boxes = torch.stack(targets).to(pred_boxes.device)

        ious = box_iou(pred_boxes, true_boxes)

        max_iou, _ = ious.max(dim=1)

        tp += (max_iou > iou_thresh).sum().item()
        fp += (max_iou <= iou_thresh).sum().item()
        fn += (len(true_boxes) - (max_iou > iou_thresh).sum().item())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mAP = precision * recall

    return mAP, precision, recall


def decode_boxes(boxes):
    """
    Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


