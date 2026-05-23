# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Make sure SG-MTF-main can find local segment_anything
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything import sam_model_registry


def _make_gn(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True, gn_groups=32):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
        self.norm = _make_gn(out_ch, gn_groups)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MedSAMMultiLevelFusion(nn.Module):
    """
    Fuse intermediate MedSAM ViT block features for classification ROI pooling.

    Input:
        block_5, block_8, block_11: [B, 768, 64, 64] for vit_b

    Output:
        cls_feat: [B, out_dim, 64, 64]
    """

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=256, gn_groups=32):
        super().__init__()

        self.proj5 = nn.Sequential(
            ConvGNAct(in_dim, hidden_dim, k=1, p=0, gn_groups=gn_groups),
            ConvGNAct(hidden_dim, hidden_dim, k=3, gn_groups=gn_groups),
        )
        self.proj8 = nn.Sequential(
            ConvGNAct(in_dim, hidden_dim, k=1, p=0, gn_groups=gn_groups),
            ConvGNAct(hidden_dim, hidden_dim, k=3, gn_groups=gn_groups),
        )
        self.proj11 = nn.Sequential(
            ConvGNAct(in_dim, hidden_dim, k=1, p=0, gn_groups=gn_groups),
            ConvGNAct(hidden_dim, hidden_dim, k=3, gn_groups=gn_groups),
        )

        self.fuse = nn.Sequential(
            ConvGNAct(hidden_dim * 3, out_dim, k=1, p=0, gn_groups=gn_groups),
            ConvGNAct(out_dim, out_dim, k=3, gn_groups=gn_groups),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        f5 = self.proj5(feats["block_5"])
        f8 = self.proj8(feats["block_8"])
        f11 = self.proj11(feats["block_11"])

        target_size = f11.shape[-2:]

        if f5.shape[-2:] != target_size:
            f5 = F.interpolate(
                f5,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if f8.shape[-2:] != target_size:
            f8 = F.interpolate(
                f8,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([f5, f8, f11], dim=1)
        x = self.fuse(x)
        return x


class MedSAMMultiScaleEncoder(nn.Module):
    """
    MedSAM image encoder + SG-MTF multi-scale adapter + intermediate block extractor.

    Original compatible output:
        f1: [B, 120, 256, 256]
        f2: [B, 240, 128, 128]
        f3: [B, 480,  64,  64]
        f4: [B, 960,  32,  32]

    New classification feature output:
        block_5:  [B, 768, 64, 64]
        block_8:  [B, 768, 64, 64]
        block_11: [B, 768, 64, 64]
        cls_feat: [B, 256, 64, 64]

    Recommended usage:
        - Use (f1, f2, f3, f4) for segmentation decoder.
        - Use cls_feat for multi-region ROI pooling and classification.
    """

    def __init__(
        self,
        checkpoint_path=None,
        model_type="vit_b",
        freeze_medsam=True,
        unfreeze_last_n=0,
        out_channels=(120, 240, 480, 960),
        gn_groups=32,
        force_img_size=1024,
        out_indices=(4, 7, 10),  # block_5, block_8, block_11
        return_cls_feat=True,
        cls_out_dim=256,
    ):
        super().__init__()

        if checkpoint_path is None:
            checkpoint_path = PROJECT_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
        checkpoint_path = str(checkpoint_path)

        self.force_img_size = force_img_size
        self.out_indices = set(out_indices)
        self.return_cls_feat = return_cls_feat

        medsam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = medsam.image_encoder

        # We do not need prompt_encoder and mask_decoder in SG-MTF backbone.
        del medsam

        if freeze_medsam:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # Optional partial fine-tuning: unfreeze last N transformer blocks.
        if unfreeze_last_n > 0:
            for p in self.image_encoder.blocks[-unfreeze_last_n:].parameters():
                p.requires_grad = True

        c1, c2, c3, c4 = out_channels

        # MedSAM neck output z: [B, 256, 64, 64]
        # Generate SG-MTF-like multi-scale features.
        self.f1_adapter = nn.Sequential(
            ConvGNAct(256, 192, k=3, s=1, gn_groups=gn_groups),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(192, 160, k=3, s=1, gn_groups=gn_groups),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(160, c1, k=3, s=1, gn_groups=gn_groups),
        )

        self.f2_adapter = nn.Sequential(
            ConvGNAct(256, 256, k=3, s=1, gn_groups=gn_groups),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(256, c2, k=3, s=1, gn_groups=gn_groups),
        )

        self.f3_adapter = nn.Sequential(
            ConvGNAct(256, 384, k=3, s=1, gn_groups=gn_groups),
            ConvGNAct(384, c3, k=3, s=1, gn_groups=gn_groups),
        )

        self.f4_adapter = nn.Sequential(
            ConvGNAct(256, 512, k=3, s=2, p=1, gn_groups=gn_groups),
            ConvGNAct(512, c4, k=3, s=1, gn_groups=gn_groups),
        )

        # For vit_b, intermediate block channels are 768.
        # If you later use vit_l/vit_h, this in_dim should be changed accordingly.
        self.cls_fusion = MedSAMMultiLevelFusion(
            in_dim=768,
            hidden_dim=256,
            out_dim=cls_out_dim,
            gn_groups=gn_groups,
        )

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input is [B,3,1024,1024].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x as [B,C,H,W], but got shape {tuple(x.shape)}")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, but got {x.shape[1]}")

        if x.shape[-2:] != (self.force_img_size, self.force_img_size):
            x = F.interpolate(
                x,
                size=(self.force_img_size, self.force_img_size),
                mode="bilinear",
                align_corners=False,
            )

        return x

    def forward_medsam_with_intermediate(self, x: torch.Tensor):
        """
        Manually run MedSAM image_encoder so that intermediate ViT blocks can be returned.

        Returns:
            z:
                MedSAM neck output, [B,256,64,64]
            inter_feats:
                block_5/block_8/block_11, each [B,768,64,64]
        """
        enc = self.image_encoder

        # patch embedding: [B,3,1024,1024] -> [B,64,64,768]
        x = enc.patch_embed(x)

        if enc.pos_embed is not None:
            x = x + enc.pos_embed

        inter_feats = {}

        for i, blk in enumerate(enc.blocks):
            x = blk(x)  # [B,H,W,C]

            if i in self.out_indices:
                # Convert to [B,C,H,W]
                inter_feats[f"block_{i + 1}"] = x.permute(0, 3, 1, 2).contiguous()

        # neck output: [B,768,64,64] -> [B,256,64,64]
        z = enc.neck(x.permute(0, 3, 1, 2).contiguous())

        return z, inter_feats

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
    ]:
        """
        Args:
            x:
                [B,1,H,W] or [B,3,H,W]
            return_dict:
                False:
                    return f1, f2, f3, f4
                    This keeps compatibility with old SG-MTF code.
                True:
                    return a dict containing:
                        seg_feats: (f1,f2,f3,f4)
                        medsam_neck: z
                        intermediate: block features
                        cls_feat: fused classification feature
        """
        x = self._preprocess_input(x)

        z, inter_feats = self.forward_medsam_with_intermediate(x)

        f1 = self.f1_adapter(z)
        f2 = self.f2_adapter(z)
        f3 = self.f3_adapter(z)
        f4 = self.f4_adapter(z)

        if not return_dict:
            return f1, f2, f3, f4

        out = {
            "seg_feats": (f1, f2, f3, f4),
            "medsam_neck": z,
            "intermediate": inter_feats,
        }

        if self.return_cls_feat:
            required = {"block_5", "block_8", "block_11"}
            missing = required - set(inter_feats.keys())
            if len(missing) > 0:
                raise RuntimeError(f"Missing intermediate features: {missing}")

            cls_feat = self.cls_fusion(inter_feats)
            out["cls_feat"] = cls_feat

        return out


if __name__ == "__main__":
    """
    Quick shape test.

    Run:
        cd D:\\pythonpro\\SG-MTF-main
        python models\\backbones\\medsam_encoder.py

    This will test:
        input image
        MedSAM neck output
        intermediate block features
        SG-MTF multi-scale features
        fused classification feature
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MedSAMMultiScaleEncoder(
        checkpoint_path=PROJECT_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth",
        model_type="vit_b",
        freeze_medsam=True,
        unfreeze_last_n=0,
        return_cls_feat=True,
    ).to(device)

    model.eval()

    # Example 1: RGB image, already 1024x1024
    x = torch.randn(1, 3, 1024, 1024).to(device)

    with torch.no_grad():
        out = model(x, return_dict=True)

    print("\n========== Input ==========")
    print("x:", tuple(x.shape))

    print("\n========== Segmentation features ==========")
    f1, f2, f3, f4 = out["seg_feats"]
    print("f1:", tuple(f1.shape))
    print("f2:", tuple(f2.shape))
    print("f3:", tuple(f3.shape))
    print("f4:", tuple(f4.shape))

    print("\n========== MedSAM features ==========")
    print("medsam_neck:", tuple(out["medsam_neck"].shape))

    for k, v in out["intermediate"].items():
        print(f"{k}:", tuple(v.shape))

    print("\n========== Classification feature ==========")
    print("cls_feat:", tuple(out["cls_feat"].shape))

    # Example 2: grayscale image, not 1024x1024
    x_gray = torch.randn(1, 1, 256, 256).to(device)

    with torch.no_grad():
        out_gray = model(x_gray, return_dict=True)

    print("\n========== Gray input resize test ==========")
    print("x_gray original:", tuple(x_gray.shape))
    print("cls_feat:", tuple(out_gray["cls_feat"].shape))
    print("seg f1:", tuple(out_gray["seg_feats"][0].shape))