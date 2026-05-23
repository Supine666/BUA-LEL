# engines/train_eval.py
from typing import Dict, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

from models.sgmtf import SGMTFModel
from utils.meters import AverageMeter
from utils.metrics_seg import dice_score_from_logits, iou_score_from_logits, seg_precision_recall_from_logits
from utils.metrics_cls import compute_classwise_sens_spec
from engines.losses import DiceLossPerSample

@torch.no_grad()
def evaluate(
    model: SGMTFModel,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    seg_thr: float,
    cls_criterion: nn.Module,
    seg_bce_weight: float,
    force_full_observed_mask: bool,
    return_preds: bool = False,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    dice_meter = AverageMeter()
    miou_meter = AverageMeter()
    segP_meter = AverageMeter()
    segR_meter = AverageMeter()

    y_true_all, y_pred_all, y_proba_all = [], [], []

    bce = nn.BCEWithLogitsLoss(reduction="none")
    dice_ps = DiceLossPerSample()

    for batch in loader:
        img, seg_gt, has_mask, c_obs, m, y, cat_targets = batch

        img = img.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        has_mask = has_mask.to(device, non_blocking=True).view(-1).float()
        c_obs = c_obs.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if force_full_observed_mask:
            m = torch.ones_like(m)

        seg_logits, cls_logits, _ = model(img, c_obs=c_obs, m=m, task="both")

        bce_map = bce(seg_logits, seg_gt)
        bce_per = bce_map.view(bce_map.size(0), -1).mean(1)
        dice_per = dice_ps(seg_logits, seg_gt)
        seg_per = seg_bce_weight * bce_per + (1.0 - seg_bce_weight) * dice_per
        Lseg = (seg_per * has_mask).sum() / has_mask.sum().clamp_min(1.0)

        Lcls = cls_criterion(cls_logits, y)
        L = Lseg + Lcls

        loss_meter.update(L.item(), img.size(0))
        seg_loss_meter.update(Lseg.item(), img.size(0))
        cls_loss_meter.update(Lcls.item(), img.size(0))

        if has_mask.sum().item() > 0:
            keep = has_mask.bool()
            dice_meter.update(dice_score_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr), int(keep.sum().item()))
            miou_meter.update(iou_score_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr), int(keep.sum().item()))
            p, r = seg_precision_recall_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr)
            segP_meter.update(p, int(keep.sum().item()))
            segR_meter.update(r, int(keep.sum().item()))

        prob = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
        pred = np.argmax(prob, axis=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred)
        y_proba_all.append(prob)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_proba = np.concatenate(y_proba_all, axis=0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    try:
        auc_macro_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_macro_ovr = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec = compute_classwise_sens_spec(cm)
    sens_macro = float(np.nanmean(sens))
    spec_macro = float(np.nanmean(spec))

    out = {
        "total_loss": loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "dice": dice_meter.avg,
        "miou": miou_meter.avg,
        "seg_precision": segP_meter.avg,
        "seg_recall": segR_meter.avg,
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "auc_macro_ovr": float(auc_macro_ovr),
        "sens_macro": sens_macro,
        "spec_macro": spec_macro,
    }
    for c in range(num_classes):
        out[f"sens_c{c}"] = float(sens[c])
        out[f"spec_c{c}"] = float(spec[c])

    if return_preds:
        out["_y_true"] = y_true
        out["_y_proba"] = y_proba

    return out

def train_one_epoch(
    model: SGMTFModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler_amp: torch.cuda.amp.GradScaler,
    loss_fn: Callable,
    max_grad_norm: float,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()

    total_meter = AverageMeter()
    seg_meter = AverageMeter()
    cls_meter = AverageMeter()

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        img, seg_gt, has_mask, c_obs, m, y, cat_targets = batch

        img = img.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        has_mask = has_mask.to(device, non_blocking=True).view(-1).float()
        c_obs = c_obs.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            Ltotal, log = loss_fn(
                model=model,
                x_img=img,
                seg_gt=seg_gt,
                has_mask=has_mask,
                y_gt=y,
                c_obs=c_obs,
                m=m,
            )
            Ltotal_scaled = Ltotal / max(1, grad_accum_steps)

        scaler_amp.scale(Ltotal_scaled).backward()

        bs = img.size(0)
        total_meter.update(float(Ltotal.detach().cpu()), bs)
        seg_meter.update(float(log.get("Lseg", 0.0)), bs)
        cls_meter.update(float(log.get("Lcls", 0.0)), bs)

        if (step % grad_accum_steps) == 0:
            scaler_amp.unscale_(optimizer)
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)

    if (len(loader) % max(1, grad_accum_steps)) != 0:
        scaler_amp.unscale_(optimizer)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        optimizer.zero_grad(set_to_none=True)

    return {"total_loss": total_meter.avg, "seg_loss": seg_meter.avg, "cls_loss": cls_meter.avg, "imp_loss": 0.0, "cons_loss": 0.0}