import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.medsam_encoder import ConvGNAct


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=32):
        super().__init__()
        self.proj = ConvGNAct(in_ch, out_ch, k=1, s=1, p=0, gn_groups=gn_groups)
        self.skip_proj = ConvGNAct(skip_ch, out_ch, k=1, s=1, p=0, gn_groups=gn_groups)
        self.fuse = nn.Sequential(
            ConvGNAct(out_ch * 2, out_ch, k=3, s=1, p=1, gn_groups=gn_groups),
            ConvGNAct(out_ch, out_ch, k=3, s=1, p=1, gn_groups=gn_groups),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        s = self.skip_proj(skip)
        out = torch.cat([x, s], dim=1)
        return self.fuse(out)


class SGMTFPerformanceDecoder(nn.Module):
    """
    UNet-like decoder with skip connections.
    output: Fseg with channels 120 at 1/4 resolution
    """

    def __init__(self, gn_groups=32):
        super().__init__()
        self.up3 = UpFuseBlock(in_ch=960, skip_ch=480, out_ch=480, gn_groups=gn_groups)
        self.up2 = UpFuseBlock(in_ch=480, skip_ch=240, out_ch=240, gn_groups=gn_groups)
        self.up1 = UpFuseBlock(in_ch=240, skip_ch=120, out_ch=120, gn_groups=gn_groups)

    def forward(self, f1, f2, f3, f4):
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        fseg = self.up1(d2, f1)
        return fseg
