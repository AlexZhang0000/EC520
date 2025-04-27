import torch
import random
import numpy as np
import torch.nn as nn

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
    """
    正确计算 mean AP, precision, recall
    """
    tp = 0
    fp = 0
    fn = 0

    for preds, targets in zip(preds_all, targets_all):
        preds = preds.sigmoid()

        obj_conf = preds[..., 4]
        class_conf = preds[..., 5:]

        pred_mask = obj_conf > 0.5

        pred_boxes = preds[pred_mask][..., :4]
        pred_boxes = decode_boxes(pred_boxes)

        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1 and targets.size(0) == 4:
                true_boxes = targets.unsqueeze(0).to(pred_boxes.device)
            else:
                true_boxes = targets.to(pred_boxes.device)
        else:
            true_boxes = torch.stack(targets).to(pred_boxes.device)

        num_gts = true_boxes.shape[0]
        num_preds = pred_boxes.shape[0]

        if num_preds == 0:
            fn += num_gts
            continue

        if num_gts == 0:
            fp += num_preds
            continue

        ious = box_iou(pred_boxes, true_boxes)  # (num_preds, num_gts)

        matched_gt = torch.full((num_gts,), False, dtype=torch.bool, device=pred_boxes.device)

        for i in range(num_preds):
            iou_per_pred = ious[i]
            max_iou, max_gt_idx = iou_per_pred.max(dim=0)

            if max_iou > iou_thresh and not matched_gt[max_gt_idx]:
                tp += 1
                matched_gt[max_gt_idx] = True
            else:
                fp += 1

        fn += (~matched_gt).sum().item()

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

class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, true_boxes):
        """
        pred_boxes: (N, 4)
        true_boxes: (N, 4)
        """
        pred_boxes = decode_boxes(pred_boxes)
        true_boxes = decode_boxes(true_boxes)

        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        true_x1, true_y1, true_x2, true_y2 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]

        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        true_w = true_x2 - true_x1
        true_h = true_y2 - true_y1

        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        true_cx = (true_x1 + true_x2) / 2
        true_cy = (true_y1 + true_y2) / 2

        inter_x1 = torch.max(pred_x1, true_x1)
        inter_y1 = torch.max(pred_y1, true_y1)
        inter_x2 = torch.min(pred_x2, true_x2)
        inter_y2 = torch.min(pred_y2, true_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = pred_w * pred_h
        true_area = true_w * true_h
        union_area = pred_area + true_area - inter_area

        iou = inter_area / (union_area + 1e-7)

        center_dist = (pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2

        enc_x1 = torch.min(pred_x1, true_x1)
        enc_y1 = torch.min(pred_y1, true_y1)
        enc_x2 = torch.max(pred_x2, true_x2)
        enc_y2 = torch.max(pred_y2, true_y2)
        enc_w = enc_x2 - enc_x1
        enc_h = enc_y2 - enc_y1
        enc_diag = enc_w ** 2 + enc_h ** 2 + 1e-7

        v = (4 / (np.pi ** 2)) * torch.pow(
            torch.atan(true_w / (true_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist / enc_diag - alpha * v

        loss = 1 - ciou
        return loss.mean()



