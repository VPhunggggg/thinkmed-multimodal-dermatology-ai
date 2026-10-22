"""
MSA_C2f — Multi-Stage Attention C2f Wrapper
=============================================
Bọc MSABlock trong kiến trúc C2f-style, nhất quán với A2C2f.

  A2C2f  : cv1 → [Sequential(ABlock,   ABlock)]   × n → cv2
  MSA_C2f: cv1 → [Sequential(MSABlock, MSABlock)] × n → cv2

YAML args: [c2, area]
  area=4 → 4 vùng không gian, mỗi vùng M token
  K=64, D=128, Z=c_ (default)

Author  : [Private]
License : [Private]
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from .msa_module import MSABlock


class MSA_C2f(nn.Module):
    """
    Multi-Stage Attention C2f Wrapper.

    Cấu trúc:
        cv1 (c1→c_) →
        [Sequential(MSABlock, MSABlock)] × n →
        cat(y[0..n]) →
        cv2 (→c2)

    y[0] = cv1(x): gradient highway — đi thẳng vào cv2, không qua MSABlock.

    Args:
        c1       (int)  : Input channels.
        c2       (int)  : Output channels.
        n        (int)  : Số Sequential block (inject bởi parse_model).
        area     (int)  : Số spatial area trong MSABlock. Mặc định: 4.
        e        (float): Channel expansion (c_ = c2 × e). Mặc định: 0.5.
        K        (int)  : Stage-1 key/query dim. Mặc định: 64.
        D        (int)  : Stage-2 context dim.   Mặc định: 128.
        mlp_ratio(float): MLP expansion ratio.   Mặc định: 2.0.

    Z tự động = c_ (full value space, không ràng buộc Z < c_).
    """

    def __init__(
        self,
        c1       : int,
        c2       : int,
        n        : int   = 1,
        area     : int   = 4,
        e        : float = 0.5,
        K        : int   = 64,
        D        : int   = 128,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()

        c_ = int(c2 * e)     # hidden channel; Z = c_ tự động

        self.cv1 = Conv(c1, c_, 1, 1)              # c1 → c_
        self.cv2 = Conv((1 + n) * c_, c2, 1)       # cat → c2

        # n × Sequential(MSABlock, MSABlock) — giống A2C2f pattern
        self.m = nn.ModuleList(
            nn.Sequential(
                MSABlock(c_, K=K, D=D, Z=c_, area=area, mlp_ratio=mlp_ratio),
                MSABlock(c_, K=K, D=D, Z=c_, area=area, mlp_ratio=mlp_ratio),
            )
            for _ in range(n)
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, c1, H, W)
        Returns:
            out: (B, c2, H, W)
        """
        y = [self.cv1(x)]                           # y[0]: gradient highway
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        c_  = self.cv1.conv.out_channels
        c2  = self.cv2.conv.out_channels
        n   = len(self.m)
        blk = self.m[0][0] if self.m else None
        if blk:
            return (
                f"c_={c_}, c2={c2}, n={n}, "
                f"K={blk.K}, D={blk.D}, Z={blk.Z}, area={blk.area}"
            )
        return f"c_={c_}, c2={c2}, n={n}"
