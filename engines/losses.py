# -*- coding: utf-8 -*-
# engines/losses.py

from typing import Callable
import torch
import torch.nn as nn

from models.sgmtf import SGMTFModel


class DiceLossPerSample(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)  # [B,1]
        return (1.0 - dice).view(dice.size(0))              # [B]


def make_nomissing_loss_fn(
    cls_criterion: nn.Module,
    seg_bce_weight: float,
    force_full_observed_mask: bool,
    use_uncertainty_weighting: bool = True,
    lambda_anchor: float = 0.01,
) -> Callable:
    """
    Loss function for SG-MTF / BUA-LEL without missingness modeling.

    Original:
        Ltotal = Lseg + Lcls

    Updated:
        1) compute segmentation loss:
            Lseg = BCE + Dice

        2) compute classification loss:
            Lcls = CE

        3) get boundary anchor loss from aux:
            Lanchor = aux["boundary_anchor_loss"]

        4) total loss:
            If use_uncertainty_weighting=True:
                use model.get_total_loss(Lseg, Lcls, Lanchor)
            Else:
                Ltotal = Lseg + Lcls + lambda_anchor * Lanchor
    """
    bce = nn.BCEWithLogitsLoss(reduction="none")
    dice_ps = DiceLossPerSample()

    def _loss_fn(
        model: SGMTFModel,
        x_img: torch.Tensor,
        seg_gt: torch.Tensor,
        has_mask: torch.Tensor,
        y_gt: torch.Tensor,
        c_obs: torch.Tensor,
        m: torch.Tensor,
        **kwargs,
    ):
        if force_full_observed_mask:
            m = torch.ones_like(m)

        # ------------------------------------------------------------
        # 1) Forward
        # ------------------------------------------------------------
        seg_logits, cls_logits, aux = model(
            x_img,
            c_obs=c_obs,
            m=m,
            task="both",
        )

        # ------------------------------------------------------------
        # 2) Segmentation loss: BCE + Dice
        # ------------------------------------------------------------
        bce_map = bce(seg_logits, seg_gt)
        bce_per = bce_map.view(bce_map.size(0), -1).mean(1)

        dice_per = dice_ps(seg_logits, seg_gt)

        seg_per = (
            seg_bce_weight * bce_per
            + (1.0 - seg_bce_weight) * dice_per
        )

        Lseg = (
            (seg_per * has_mask).sum()
            / has_mask.sum().clamp_min(1.0)
        )

        # ------------------------------------------------------------
        # 3) Classification loss
        # ------------------------------------------------------------
        Lcls = cls_criterion(cls_logits, y_gt)

        # ------------------------------------------------------------
        # 4) Boundary anchor loss
        # ------------------------------------------------------------
        Lanchor = aux.get("boundary_anchor_loss", None)

        if Lanchor is None:
            Lanchor = torch.zeros(
                (),
                device=x_img.device,
                dtype=Lseg.dtype,
            )

        # ------------------------------------------------------------
        # 5) Total loss
        # ------------------------------------------------------------
        if use_uncertainty_weighting and hasattr(model, "get_total_loss"):
            Ltotal, weights = model.get_total_loss(
                Lseg=Lseg,
                Lcls=Lcls,
                Limp=0.0,
                Lcons=0.0,
                Lanchor=Lanchor,
                lambda_anchor=lambda_anchor,
            )
        else:
            Ltotal = Lseg + Lcls + float(lambda_anchor) * Lanchor
            weights = {
                "w_seg": 1.0,
                "w_cls": 1.0,
                "w_imp": 0.0,
                "lambda_cons": 0.0,
                "lambda_anchor": float(lambda_anchor),
            }

        # ------------------------------------------------------------
        # 6) Logging
        # ------------------------------------------------------------
        log = {
            "Lseg": float(Lseg.detach().cpu()),
            "Lcls": float(Lcls.detach().cpu()),
            "Limp": 0.0,
            "Lcons": 0.0,
            "Lanchor": float(Lanchor.detach().cpu()),
            "Ltotal": float(Ltotal.detach().cpu()),

            "w_seg": float(weights.get("w_seg", 1.0)),
            "w_cls": float(weights.get("w_cls", 1.0)),
            "w_imp": float(weights.get("w_imp", 0.0)),
            "lambda_cons": float(weights.get("lambda_cons", 0.0)),
            "lambda_anchor": float(weights.get("lambda_anchor", lambda_anchor)),
        }

        return Ltotal, log

    return _loss_fn