import torch
import numpy as np

# --- 计算 per-class IoU 和 mean IoU ---

def compute_iou(preds, labels, num_classes):
    """
    Args:
        preds: (N, H, W) 预测标签
        labels: (N, H, W) 真实标签
        num_classes: 类别数
    Returns:
        ious: 每一类的IoU (array)
        miou: 所有类平均的mean IoU (float)
    """
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls

        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection

        if union == 0:
            iou = float('nan')  # 如果这个类不存在，置为nan
        else:
            iou = intersection / union

        ious.append(iou)

    # 只对有意义的类别取平均（去掉nan）
    ious = np.array(ious)
    miou = np.nanmean(ious)

    return ious, miou

# --- 把IoU打印成表格格式 ---
def print_iou(ious, class_names=None):
    """
    打印每一类的IoU
    Args:
        ious: list or array of IoUs
        class_names: list of class names
    """
    print("\nPer-Class IoU:")
    for idx, iou in enumerate(ious):
        if class_names:
            print(f"{class_names[idx]}: {iou:.4f}")
        else:
            print(f"Class {idx}: {iou:.4f}")

    print("\n")
