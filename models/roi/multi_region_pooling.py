# -*- coding: utf-8 -*-
"""
Multi-region ROI pooling for SG-MTF / BUA-LEL with MedSAM features.

Input:
    feat_map:
        [B, C, H, W], e.g. cls_feat = [B,256,64,64]

    roi_prob:
        [B, 1, H0, W0], e.g. sigmoid(seg_logits) = [B,1,256,256]

Key design:
    1. Region masks are constructed at roi_prob resolution.
    2. Region masks are then resized to feat_map resolution.
    3. Masked statistics pooling is performed on feat_map: mean + max + std.

Updated design:
    1. less-overlap peritumor:
        peritumor = dilate_large(mask) - dilate_small(mask)

    2. lesion-size-aware kernels:
        For each sample, estimate equivalent lesion radius:
            r = sqrt(area / pi)
        Then choose adaptive boundary/peritumor kernels.

    3. small-lesion peritumor suppression:
        For extremely small lesions, peritumor is unreliable and may contain
        background noise. Therefore, a size-aware gate suppresses peritumor
        features and/or peritumor region attention logits.

    4. detach_roi:
        The segmentation probability map can be detached before morphology
        decomposition to prevent classification gradients from disturbing
        the segmentation decoder.

    5. residual global fusion:
        The global representation is used as a residual pathway to stabilize
        classification, especially when region masks are noisy.

Enhanced design:
    Region branch:
        core/boundary/peritumor mask
        -> masked mean/max/std
        -> shared region projection
        -> region attention
        -> v_region [B,C]

    Global branch:
        feat_map
        -> global mean/max/std
        -> global projection
        -> v_global [B,C]

    Final:
        concat(v_region, v_global)
        -> final projection
        -> residual add v_global
        -> v_img [B,C]
"""

from typing import Dict, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize_mask(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Clamp mask into [0,1].
    """
    mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = mask.clamp(min=0.0, max=1.0)
    return mask


def _make_odd_kernel(k: int, min_k: int = 1) -> int:
    """
    Ensure kernel size is odd and >= min_k.
    """
    k = int(max(k, min_k))
    if k % 2 == 0:
        k += 1
    return k


def soft_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Differentiable dilation using max pooling.

    Args:
        mask: [B, 1, H, W]
        kernel_size: odd integer
    """
    if kernel_size <= 1:
        return mask

    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def soft_erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Differentiable erosion using negative max pooling.

    Args:
        mask: [B, 1, H, W]
        kernel_size: odd integer
    """
    if kernel_size <= 1:
        return mask

    pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


class MaskedStatsPool2d(nn.Module):
    """
    Masked statistics pooling.

    Compared with masked average pooling, this keeps richer classification
    cues from each region:

        mean: stable region-level semantic response
        max:  strongest local response inside the region
        std:  intra-region heterogeneity / texture variation

    feat_map: [B, C, H, W]
    mask:     [B, 1, H, W]
    output:   [B, 3C] = concat(mean, max, std)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map should be [B,C,H,W], got {tuple(feat_map.shape)}")

        if mask.dim() != 4:
            raise ValueError(f"mask should be [B,1,H,W], got {tuple(mask.shape)}")

        if mask.shape[1] != 1:
            raise ValueError(f"mask channel should be 1, got {mask.shape[1]}")

        if mask.shape[-2:] != feat_map.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=feat_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        mask = _safe_normalize_mask(mask)
        B, C, H, W = feat_map.shape

        # -----------------------------
        # 1) Masked weighted mean
        # -----------------------------
        denominator = mask.sum(dim=(2, 3)).clamp_min(self.eps)  # [B,1]
        weighted_feat = feat_map * mask
        mean = weighted_feat.sum(dim=(2, 3)) / denominator      # [B,C]

        # -----------------------------
        # 2) Masked weighted std
        # -----------------------------
        var = (((feat_map - mean.view(B, C, 1, 1)) ** 2) * mask).sum(dim=(2, 3)) / denominator
        std = torch.sqrt(var.clamp_min(self.eps))              # [B,C]

        # -----------------------------
        # 3) Masked max
        # -----------------------------
        # For soft masks, locations with very small weights are ignored.
        valid = mask > self.eps                                # [B,1,H,W]
        valid_expand = valid.expand_as(feat_map)               # [B,C,H,W]

        neg_inf = torch.finfo(feat_map.dtype).min
        masked_feat = feat_map.masked_fill(~valid_expand, neg_inf)
        max_val = masked_feat.flatten(2).max(dim=2).values      # [B,C]

        # If a region is empty after resizing/interpolation, avoid -inf.
        has_valid = valid.flatten(2).any(dim=2)                 # [B,1]
        max_val = torch.where(has_valid.expand_as(max_val), max_val, torch.zeros_like(max_val))

        return torch.cat([mean, max_val, std], dim=1)


class GlobalStatsPool2d(nn.Module):
    """
    Global statistics pooling over the whole feature map.

    feat_map: [B, C, H, W]
    output:   [B, 3C] = concat(global_mean, global_max, global_std)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map should be [B,C,H,W], got {tuple(feat_map.shape)}")

        mean = feat_map.mean(dim=(2, 3))
        max_val = feat_map.flatten(2).max(dim=2).values
        std = torch.sqrt(feat_map.var(dim=(2, 3), unbiased=False).clamp_min(self.eps))

        return torch.cat([mean, max_val, std], dim=1)


class RegionAttentionFusion(nn.Module):
    """
    Fuse core/boundary/peritumor features by learnable region attention.

    Input:
        region_feats: [B, 3, C]

    Optional:
        region_logit_bias: [B,3]
            Additive bias before softmax.
            Used to suppress unreliable peritumor for small lesions.

    Output:
        v_img:   [B, C]
        weights: [B, 3]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        region_feats: torch.Tensor,
        region_logit_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if region_feats.dim() != 3:
            raise ValueError(
                f"region_feats should be [B,3,C], got {tuple(region_feats.shape)}"
            )

        logits = self.score(region_feats).squeeze(-1)  # [B,3]

        if region_logit_bias is not None:
            if region_logit_bias.shape != logits.shape:
                raise ValueError(
                    f"region_logit_bias shape should be {tuple(logits.shape)}, "
                    f"got {tuple(region_logit_bias.shape)}"
                )
            logits = logits + region_logit_bias

        weights = torch.softmax(logits, dim=1)         # [B,3]
        v_img = (region_feats * weights.unsqueeze(-1)).sum(dim=1)

        return v_img, weights


class MultiRegionROIPooling(nn.Module):
    """
    Enhanced core-boundary-peritumor ROI pooling.

    Region definitions:
        core:
            ROI probability map.

        boundary:
            dilate_small(ROI) - erode_small(ROI)

        peritumor:
            less-overlap version:
                dilate_large(ROI) - dilate_small(ROI)

    Adaptive kernels:
        Estimate equivalent radius:
            r = sqrt(area / pi)

        boundary_radius = clamp(boundary_ratio * r, min_boundary_radius, max_boundary_radius)
        peritumor_radius = clamp(peritumor_ratio * r, min_peritumor_radius, max_peritumor_radius)

        kernel = 2 * radius + 1

    Small lesion protection:
        If lesion area is small, peritumor is likely to contain background.
        We compute a peritumor reliability gate:
            gate = sigmoid((area - threshold) / temperature)
        Then:
            f_peritumor = gate * f_peritumor
        And optionally:
            logit_peritumor += log(gate)
        so attention will naturally reduce peritumor contribution.
    """

    def __init__(
        self,
        in_channels: int = 256,

        # Fallback fixed kernels.
        # Used when use_adaptive_kernel=False.
        boundary_kernel: int = 7,
        peritumor_kernel: int = 17,

        # Region fusion.
        use_region_attention: bool = True,
        dropout: float = 0.1,
        eps: float = 1e-6,

        # ROI processing.
        binarize_roi: bool = False,
        binarize_threshold: float = 0.5,

        # NEW: prevent classification gradients from disturbing segmentation mask.
        detach_roi: bool = True,

        # 1. less-overlap peritumor
        less_overlap_peritumor: bool = True,

        # 2. lesion-size-aware adaptive kernel
        use_adaptive_kernel: bool = True,
        boundary_radius_ratio: float = 0.15,
        peritumor_radius_ratio: float = 0.40,
        min_boundary_radius: int = 2,
        max_boundary_radius: int = 4,
        min_peritumor_radius: int = 4,
        max_peritumor_radius: int = 8,

        # 3. small-lesion peritumor suppression
        use_small_lesion_gate: bool = True,
        small_lesion_area_threshold: float = 500.0,
        small_lesion_gate_temperature: float = 150.0,
        hard_disable_peritumor_for_small: bool = False,
        hard_small_lesion_area_threshold: float = 300.0,

        # Whether to add log(gate) as attention logit bias for peritumor.
        suppress_peritumor_attention: bool = True,
    ):
        super().__init__()

        if boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel should be an odd integer.")
        if peritumor_kernel % 2 == 0:
            raise ValueError("peritumor_kernel should be an odd integer.")
        if peritumor_kernel < boundary_kernel:
            raise ValueError("peritumor_kernel should be >= boundary_kernel.")

        self.in_channels = int(in_channels)
        self.boundary_kernel = int(boundary_kernel)
        self.peritumor_kernel = int(peritumor_kernel)
        self.use_region_attention = bool(use_region_attention)
        self.eps = float(eps)

        self.binarize_roi = bool(binarize_roi)
        self.binarize_threshold = float(binarize_threshold)
        self.detach_roi = bool(detach_roi)

        self.less_overlap_peritumor = bool(less_overlap_peritumor)

        self.use_adaptive_kernel = bool(use_adaptive_kernel)
        self.boundary_radius_ratio = float(boundary_radius_ratio)
        self.peritumor_radius_ratio = float(peritumor_radius_ratio)
        self.min_boundary_radius = int(min_boundary_radius)
        self.max_boundary_radius = int(max_boundary_radius)
        self.min_peritumor_radius = int(min_peritumor_radius)
        self.max_peritumor_radius = int(max_peritumor_radius)

        self.use_small_lesion_gate = bool(use_small_lesion_gate)
        self.small_lesion_area_threshold = float(small_lesion_area_threshold)
        self.small_lesion_gate_temperature = float(small_lesion_gate_temperature)
        self.hard_disable_peritumor_for_small = bool(hard_disable_peritumor_for_small)
        self.hard_small_lesion_area_threshold = float(hard_small_lesion_area_threshold)
        self.suppress_peritumor_attention = bool(suppress_peritumor_attention)

        # ---------------------------------------------------------
        # Enhanced statistics pooling
        # ---------------------------------------------------------
        # Region branch:
        #   masked mean/max/std -> [B,3C] -> region_proj -> [B,C]
        self.pool = MaskedStatsPool2d(eps=eps)
        self.region_proj = nn.Sequential(
            nn.LayerNorm(in_channels * 3),
            nn.Linear(in_channels * 3, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Global branch:
        #   global mean/max/std -> [B,3C] -> global_proj -> [B,C]
        self.global_pool = GlobalStatsPool2d(eps=eps)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(in_channels * 3),
            nn.Linear(in_channels * 3, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_region_attention:
            self.region_fusion = RegionAttentionFusion(
                in_dim=in_channels,
                hidden_dim=max(64, in_channels // 2),
                dropout=dropout,
            )
        else:
            self.region_fusion = None

        # Final fusion:
        #   concat(v_region, v_global) -> [B,2C] -> final_proj -> [B,C]
        #   then residual add v_global -> [B,C]
        self.final_proj = nn.Sequential(
            nn.LayerNorm(in_channels * 2),
            nn.Linear(in_channels * 2, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _prepare_roi(self, roi_prob: torch.Tensor) -> torch.Tensor:
        """
        Normalize roi_prob and optionally binarize it.

        If detach_roi=True, the classification branch uses the predicted
        segmentation probability map only as a spatial prior, and its gradients
        will not flow back to the segmentation decoder through morphology masks.
        """
        if self.detach_roi:
            roi_prob = roi_prob.detach()

        roi = _safe_normalize_mask(roi_prob)

        if self.binarize_roi:
            roi = (roi > self.binarize_threshold).to(dtype=roi.dtype)

        return roi

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize a soft region mask to target_size.
        """
        if mask.shape[-2:] == target_size:
            return mask

        return F.interpolate(
            mask,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def _compute_area_radius_and_gate(self, roi: torch.Tensor):
        """
        Args:
            roi: [B,1,H,W]

        Returns:
            lesion_area: [B]
            equiv_radius: [B]
            peritumor_gate: [B]
        """
        B = roi.size(0)
        flat = roi.view(B, -1)

        # Soft area from predicted probability mask.
        # If binarize_roi=True, this becomes binary area.
        lesion_area = flat.sum(dim=1).clamp_min(1.0)  # [B]

        equiv_radius = torch.sqrt(lesion_area / math.pi)  # [B]

        if self.use_small_lesion_gate:
            temp = max(self.small_lesion_gate_temperature, 1e-6)
            peritumor_gate = torch.sigmoid(
                (lesion_area - self.small_lesion_area_threshold) / temp
            )
        else:
            peritumor_gate = torch.ones_like(lesion_area)

        if self.hard_disable_peritumor_for_small:
            hard_mask = (lesion_area >= self.hard_small_lesion_area_threshold).to(roi.dtype)
            peritumor_gate = peritumor_gate * hard_mask

        peritumor_gate = peritumor_gate.clamp(0.0, 1.0)

        return lesion_area, equiv_radius, peritumor_gate

    def _adaptive_kernels_from_radius(self, radius_value: float) -> Tuple[int, int]:
        """
        Convert equivalent lesion radius to adaptive odd kernels.

        radius_value:
            scalar equivalent radius in pixels at roi_prob resolution.
        """
        if not self.use_adaptive_kernel:
            return self.boundary_kernel, self.peritumor_kernel

        b_radius = int(round(self.boundary_radius_ratio * radius_value))
        p_radius = int(round(self.peritumor_radius_ratio * radius_value))

        b_radius = max(self.min_boundary_radius, min(self.max_boundary_radius, b_radius))
        p_radius = max(self.min_peritumor_radius, min(self.max_peritumor_radius, p_radius))

        # Ensure peritumor radius > boundary radius.
        if p_radius <= b_radius:
            p_radius = min(self.max_peritumor_radius, b_radius + 1)

        b_kernel = _make_odd_kernel(2 * b_radius + 1, min_k=3)
        p_kernel = _make_odd_kernel(2 * p_radius + 1, min_k=b_kernel + 2)

        return b_kernel, p_kernel

    def build_region_masks(
        self,
        roi_prob: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Build core, boundary, and peritumor masks.

        Morphological operations are performed at roi_prob resolution.
        Adaptive kernels are computed per sample according to pred-mask area.
        """
        if roi_prob.dim() != 4:
            raise ValueError(f"roi_prob should be [B,1,H,W], got {tuple(roi_prob.shape)}")

        if roi_prob.shape[1] != 1:
            raise ValueError(f"roi_prob channel should be 1, got {roi_prob.shape[1]}")

        roi = self._prepare_roi(roi_prob)
        B = roi.size(0)

        lesion_area, equiv_radius, peritumor_gate = self._compute_area_radius_and_gate(roi)

        core_hr_list: List[torch.Tensor] = []
        boundary_hr_list: List[torch.Tensor] = []
        peritumor_hr_list: List[torch.Tensor] = []

        boundary_kernel_list = []
        peritumor_kernel_list = []

        # Per-sample adaptive morphology.
        for i in range(B):
            roi_i = roi[i:i + 1]  # [1,1,H,W]
            r_i = float(equiv_radius[i].detach().cpu().item())

            b_kernel, p_kernel = self._adaptive_kernels_from_radius(r_i)
            boundary_kernel_list.append(b_kernel)
            peritumor_kernel_list.append(p_kernel)

            core_i = roi_i

            dil_b_i = soft_dilate(roi_i, b_kernel)
            ero_b_i = soft_erode(roi_i, b_kernel)
            boundary_i = (dil_b_i - ero_b_i).clamp(0.0, 1.0)

            dil_p_i = soft_dilate(roi_i, p_kernel)

            if self.less_overlap_peritumor:
                # Less-overlap annular peritumor:
                # outer ring between large dilation and small dilation.
                peritumor_i = (dil_p_i - dil_b_i).clamp(0.0, 1.0)
            else:
                # Original version:
                # the whole exterior region outside ROI.
                peritumor_i = (dil_p_i - roi_i).clamp(0.0, 1.0)

            # Size-aware reliability gate.
            gate_i = peritumor_gate[i].view(1, 1, 1, 1).to(dtype=peritumor_i.dtype)
            peritumor_i = peritumor_i * gate_i

            core_hr_list.append(core_i)
            boundary_hr_list.append(boundary_i)
            peritumor_hr_list.append(peritumor_i)

        core_hr = torch.cat(core_hr_list, dim=0)
        boundary_hr = torch.cat(boundary_hr_list, dim=0)
        peritumor_hr = torch.cat(peritumor_hr_list, dim=0)

        core = self._resize_mask(core_hr, target_size)
        boundary = self._resize_mask(boundary_hr, target_size)
        peritumor = self._resize_mask(peritumor_hr, target_size)

        core = _safe_normalize_mask(core)
        boundary = _safe_normalize_mask(boundary)
        peritumor = _safe_normalize_mask(peritumor)

        info = {
            "core": core,
            "boundary": boundary,
            "peritumor": peritumor,

            "core_hr": core_hr,
            "boundary_hr": boundary_hr,
            "peritumor_hr": peritumor_hr,

            # Debug metadata.
            "lesion_area": lesion_area.detach(),
            "equiv_radius": equiv_radius.detach(),
            "peritumor_gate": peritumor_gate.detach(),
            "boundary_kernel": torch.tensor(
                boundary_kernel_list,
                device=roi.device,
                dtype=torch.long,
            ),
            "peritumor_kernel": torch.tensor(
                peritumor_kernel_list,
                device=roi.device,
                dtype=torch.long,
            ),
        }

        return info

    def _build_region_logit_bias(
        self,
        peritumor_gate: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build attention logit bias for [core, boundary, peritumor].

        If peritumor_gate is small, log(gate) is a negative value and will
        reduce peritumor attention probability after softmax.
        """
        B = peritumor_gate.size(0)

        bias = torch.zeros(B, 3, dtype=dtype, device=device)

        if self.suppress_peritumor_attention:
            gate = peritumor_gate.to(device=device, dtype=dtype).clamp_min(self.eps)
            bias[:, 2] = torch.log(gate)

        return bias

    def forward(
        self,
        feat_map: torch.Tensor,
        roi_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feat_map:
                [B,C,H,W], e.g. [B,256,64,64]

            roi_prob:
                [B,1,H0,W0], e.g. [B,1,256,256]

        Returns:
            v_img:
                [B,C]

            info:
                f_core, f_boundary, f_peritumor, region_feats, region_weights,
                v_region, v_global, v_fused, raw statistics features,
                and region masks / adaptive metadata.
        """
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map should be [B,C,H,W], got {tuple(feat_map.shape)}")

        B, C, H, W = feat_map.shape

        if C != self.in_channels:
            raise ValueError(
                f"feat_map channel mismatch: expected {self.in_channels}, got {C}"
            )

        masks = self.build_region_masks(roi_prob, target_size=(H, W))

        # -----------------------------------------------------
        # 1. Region branch: masked mean/max/std -> proj
        # -----------------------------------------------------
        f_core_stats = self.pool(feat_map, masks["core"])            # [B,3C]
        f_boundary_stats = self.pool(feat_map, masks["boundary"])    # [B,3C]
        f_peritumor_stats = self.pool(feat_map, masks["peritumor"])  # [B,3C]

        f_core = self.region_proj(f_core_stats)              # [B,C]
        f_boundary = self.region_proj(f_boundary_stats)      # [B,C]
        f_peritumor = self.region_proj(f_peritumor_stats)    # [B,C]

        # Feature-level peritumor suppression.
        # The mask itself has already been gated, but this further prevents
        # small-lesion peritumor features from dominating through attention.
        peritumor_gate = masks["peritumor_gate"].to(
            device=feat_map.device,
            dtype=feat_map.dtype,
        ).view(B, 1)

        if self.use_small_lesion_gate:
            f_peritumor = f_peritumor * peritumor_gate

        region_feats = torch.stack(
            [f_core, f_boundary, f_peritumor],
            dim=1,
        )  # [B,3,C]

        if self.use_region_attention:
            region_logit_bias = self._build_region_logit_bias(
                peritumor_gate=masks["peritumor_gate"],
                dtype=feat_map.dtype,
                device=feat_map.device,
            )
            v_region, region_weights = self.region_fusion(
                region_feats,
                region_logit_bias=region_logit_bias,
            )  # [B,C], [B,3]
        else:
            # Average fusion, but still use gate-aware weighting.
            base_weights = torch.ones(B, 3, device=feat_map.device, dtype=feat_map.dtype)
            if self.use_small_lesion_gate:
                base_weights[:, 2] = peritumor_gate.view(-1)
            base_weights = base_weights / base_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

            v_region = (region_feats * base_weights.unsqueeze(-1)).sum(dim=1)  # [B,C]
            region_weights = base_weights

        # -----------------------------------------------------
        # 2. Global branch: global mean/max/std -> proj
        # -----------------------------------------------------
        global_stats = self.global_pool(feat_map)     # [B,3C]
        v_global = self.global_proj(global_stats)     # [B,C]

        # -----------------------------------------------------
        # 3. Final fusion: concat(v_region, v_global) -> residual v_img
        # -----------------------------------------------------
        v_fused = self.final_proj(
            torch.cat([v_region, v_global], dim=1)
        )  # [B,C]

        # NEW: residual global fusion.
        # This preserves a stable image-level pathway while injecting
        # region-aware morphology information.
        v_img = v_fused + v_global                    # [B,C]

        info = {
            "f_core": f_core,
            "f_boundary": f_boundary,
            "f_peritumor": f_peritumor,
            "f_core_stats": f_core_stats,
            "f_boundary_stats": f_boundary_stats,
            "f_peritumor_stats": f_peritumor_stats,
            "region_feats": region_feats,
            "region_weights": region_weights,
            "v_region": v_region,
            "global_stats": global_stats,
            "v_global": v_global,
            "v_fused": v_fused,
            "v_img": v_img,
            "masks": masks,
        }

        return v_img, info


if __name__ == "__main__":
    """
    Quick shape test.

    Run:
        cd D:\\pythonpro\\SG-MTF-main
        python models\\roi\\multi_region_pooling.py
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    C = 256

    cls_feat = torch.randn(B, C, 64, 64).to(device)

    # Simulate one tiny lesion and one larger lesion.
    roi_prob = torch.zeros(B, 1, 256, 256).to(device)
    roi_prob[0, 0, 125:135, 125:135] = 1.0
    roi_prob[1, 0, 80:170, 90:180] = 1.0

    pool = MultiRegionROIPooling(
        in_channels=256,
        boundary_kernel=7,
        peritumor_kernel=17,
        use_region_attention=True,
        less_overlap_peritumor=True,
        use_adaptive_kernel=True,
        use_small_lesion_gate=True,
        detach_roi=True,
    ).to(device)

    v_img, info = pool(cls_feat, roi_prob)

    print("========== Inputs ==========")
    print("cls_feat:", tuple(cls_feat.shape))
    print("roi_prob:", tuple(roi_prob.shape))

    print("\n========== Adaptive metadata ==========")
    print("lesion_area:", info["masks"]["lesion_area"].detach().cpu().numpy())
    print("equiv_radius:", info["masks"]["equiv_radius"].detach().cpu().numpy())
    print("peritumor_gate:", info["masks"]["peritumor_gate"].detach().cpu().numpy())
    print("boundary_kernel:", info["masks"]["boundary_kernel"].detach().cpu().numpy())
    print("peritumor_kernel:", info["masks"]["peritumor_kernel"].detach().cpu().numpy())

    print("\n========== Region masks at feature-map resolution ==========")
    for k in ["core", "boundary", "peritumor"]:
        v = info["masks"][k]
        print(f"{k}:", tuple(v.shape), "sum:", float(v.sum()))

    print("\n========== Region masks at original ROI resolution ==========")
    for k in ["core_hr", "boundary_hr", "peritumor_hr"]:
        v = info["masks"][k]
        print(f"{k}:", tuple(v.shape), "sum:", float(v.sum()))

    print("\n========== Region features ==========")
    print("f_core_stats:", tuple(info["f_core_stats"].shape))
    print("f_boundary_stats:", tuple(info["f_boundary_stats"].shape))
    print("f_peritumor_stats:", tuple(info["f_peritumor_stats"].shape))
    print("f_core:", tuple(info["f_core"].shape))
    print("f_boundary:", tuple(info["f_boundary"].shape))
    print("f_peritumor:", tuple(info["f_peritumor"].shape))
    print("region_feats:", tuple(info["region_feats"].shape))

    print("\n========== Global branch ==========")
    print("global_stats:", tuple(info["global_stats"].shape))
    print("v_global:", tuple(info["v_global"].shape))

    print("\n========== Fusion output ==========")
    print("region_weights:", tuple(info["region_weights"].shape))
    print("region_weights:", info["region_weights"].detach().cpu().numpy())
    print("v_region:", tuple(info["v_region"].shape))
    print("v_fused:", tuple(info["v_fused"].shape))
    print("v_img:", tuple(v_img.shape))