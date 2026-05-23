# utils/optim.py
from typing import Tuple
import torch.nn as nn

def freeze_bn_running_stats(model: nn.Module):
    """冻结 BN 的 running mean/var，减少小 batch 波动（不会冻结权重）"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

def build_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    cls_lr_mult: float,
    weight_decay: float,
    cls_name_keywords: Tuple[str, ...],
):
    cls_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(k in lname for k in cls_name_keywords):
            cls_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if cls_params:
        groups.append({"params": cls_params, "lr": base_lr * cls_lr_mult, "weight_decay": weight_decay})

    if not groups:
        groups = [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    return groups, (len(other_params), len(cls_params))