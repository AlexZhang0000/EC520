import torch
import torch.nn as nn
import random
import numpy as np

def box_iou(box1, box2):
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  

    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou

def decode_boxes(boxes):
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def compute_map(preds_all, targets_all, iou_thresh=0.5):
    tp = 0
    fp = 0
    fn = 0

    for preds, targets in zip(preds_all, targets_all):
        preds = preds.sigmoid()

        obj_conf = preds[..., 4]
        class_conf = preds[..., 5:]

        pred_mask = obj_conf > 0.3
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

        ious = box_iou(pred_boxes, true_boxes)

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

class SimpleBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_boxes, true_boxes):
        return self.mse(pred_boxes, true_boxes)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



