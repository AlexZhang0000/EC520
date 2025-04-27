import torch

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: (N, 4)
    box2: (M, 4)
    """
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:,:,0] * wh[:,:,1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou  # (N, M)

def compute_map(preds_all, targets_all, iou_thresh=0.5):
    """
    计算 mean AP, precision, recall
    preds_all: list of tensors, 每个是(batch_size, N, 5+num_classes)
    targets_all: list of targets, 每个是boxes (list of [x1,y1,x2,y2])
    """
    tp = 0
    fp = 0
    fn = 0

    for preds, targets in zip(preds_all, targets_all):
        preds = preds.sigmoid()

        obj_conf = preds[...,4]
        class_conf = preds[...,5:]

        # 取每个预测置信度>0.5且属于某个类别
        pred_mask = obj_conf > 0.5

        if pred_mask.sum() == 0:
            fn += len(targets)
            continue

        pred_boxes = preds[pred_mask][..., :4]  # 只取位置
        pred_boxes = decode_boxes(pred_boxes)  # 从(cx,cy,w,h)恢复(x1,y1,x2,y2)

        if len(targets) == 0:
            fp += pred_boxes.size(0)
            continue

        true_boxes = torch.stack(targets).to(pred_boxes.device)

        ious = box_iou(pred_boxes, true_boxes)

        max_iou, _ = ious.max(dim=1)

        tp += (max_iou > iou_thresh).sum().item()
        fp += (max_iou <= iou_thresh).sum().item()
        fn += (len(true_boxes) - (max_iou > iou_thresh).sum().item())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mAP = precision * recall  # 简化版示意

    return mAP, precision, recall

def decode_boxes(boxes):
    """
    (cx, cy, w, h) --> (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)
