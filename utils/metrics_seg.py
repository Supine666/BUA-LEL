import torch
import torch.nn.functional as F


def _as_probs(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if x.min().item() < 0.0 or x.max().item() > 1.0:
        return torch.sigmoid(x)
    return x


def dice_score_from_probs(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_probs(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def seg_precision_recall_from_probs(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.mean().item(), recall.mean().item()


def dice_score_from_logits(logits, targets, thr=0.5, eps=1e-6):
    return dice_score_from_probs(torch.sigmoid(logits), targets, thr=thr, eps=eps)


def iou_score_from_logits(logits, targets, thr=0.5, eps=1e-6):
    return iou_score_from_probs(torch.sigmoid(logits), targets, thr=thr, eps=eps)


def seg_precision_recall_from_logits(logits, targets, thr=0.5, eps=1e-6):
    return seg_precision_recall_from_probs(torch.sigmoid(logits), targets, thr=thr, eps=eps)


def mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp_min(0.0)


def boundary_f1_score(pred_prob: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred_prob = _as_probs(pred_prob)
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target_mask > 0.5).float()
    pred_boundary = mask_boundary(pred_bin)
    target_boundary = mask_boundary(target_bin)
    tp = (pred_boundary * target_boundary).sum(dim=(2, 3))
    fp = (pred_boundary * (1.0 - target_boundary)).sum(dim=(2, 3))
    fn = ((1.0 - pred_boundary) * target_boundary).sum(dim=(2, 3))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    return f1.mean().item()
