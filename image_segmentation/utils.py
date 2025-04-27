import torch

def compute_iou(preds, masks, num_classes):
    ious = []
    preds = preds.view(-1)
    masks = masks.view(-1)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)

    miou = np.nanmean(ious)
    return ious, miou
