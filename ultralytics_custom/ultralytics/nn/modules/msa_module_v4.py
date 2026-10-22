"""
MSABlock — Multi-Stage Attention Block  (3-stage redesign)
===========================================================

Kiến trúc 3-stage thay cho 4-stage cũ:

  Stage 1 — K-space coarse attention     "where to look"
    · L2-normalize K1, K2  → dot product ∈ [-1,1], loại overflow FP16
    · Learnable temperature (khởi tạo 0.5) thay fixed /√K
    · nan_to_num + softmax

  Stage 2 — D-space feature aggregation  "what context"
    · Dùng attention map (out1) để aggregate D-features
    · LayerNorm

  Stage 3 — Gated Z-space value          "what to output"
    · SwiGLU gate: silu(Linear(D→Z)) × zz
    · hidden từ D-space kiểm soát bao nhiêu Z-value đi qua
    · LayerNorm

So với 4-stage cũ:
  · Bỏ Stage 4 (out3@out3ᵀ — nguồn NaN chính)
  · Bỏ Stage 2's second M×M (hidden@D2)
  · M×M operations: 3× → 1×
  · Thêm SwiGLU gate — expressive, bounded
  · L2-norm thay clamp — stable, không cần hyperparameter

Sub-block 2 (MLP) và Layer Scale giữ nguyên.

Author  : [Private]
License : [Private]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSABlock(nn.Module):
    """
    Multi-Stage Attention Block (3-stage, NaN-safe).

    Sub-block 1: LN → 3-stage Area Attention → Layer Scale → residual
    Sub-block 2: LN → MLP (GELU)             → Layer Scale → residual

    Args:
        c        (int)  : Input/output channels.
        K        (int)  : Stage-1 key/query dim (K1, K2). Mặc định: 64.
        D        (int)  : Stage-2 context dim  (D1).      Mặc định: 128.
        Z        (int)  : Stage-3 value dim    (Z).       Mặc định: c.
        area     (int)  : Số spatial area (phải là số chính phương). Mặc định: 4.
        mlp_ratio(float): MLP expansion ratio.            Mặc định: 2.0.
    """

    def __init__(
        self,
        c        : int,
        K        : int        = 64,
        D        : int        = 128,
        Z        : int | None = None,
        area     : int        = 4,
        mlp_ratio: float      = 2.0,
    ):
        super().__init__()
        Z = Z if Z is not None else c    # Z = c: full value space

        self.c, self.K, self.D, self.Z = c, K, D, Z
        self.area = area

        # ── SUB-BLOCK 1: Area Attention ───────────────────────────────────

        self.norm1 = nn.LayerNorm(c)

        # Conv1×1 toàn cục — shared weights cho mọi area
        # proj_d2 đã bỏ (không còn Stage 2 M×M)
        self.proj_k1  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_k2  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_d1  = nn.Conv2d(c, D, 1, bias=False)
        self.proj_z   = nn.Conv2d(c, Z, 1, bias=False)
        self.proj_out = nn.Conv2d(Z, c, 1, bias=False)

        # Normalization trung gian
        self.norm_hidden = nn.LayerNorm(D)
        self.norm_z      = nn.LayerNorm(Z)

        # Learnable temperature — khởi tạo 0.5, clamp ≥ 0.1
        # Thay cho fixed /√K cũ — tự adapt trong quá trình train
        self.temperature = nn.Parameter(torch.tensor(0.5))

        # SwiGLU gate: Linear(D→Z) — hidden (D-space) gating Z-values
        self.dz_gate = nn.Linear(D, Z, bias=False)

        # Layer Scale — quan trọng cho training stability
        # khởi tạo 0.01: attention path nhỏ ban đầu, residual chiếm ưu thế
        self.gamma1 = nn.Parameter(1e-2 * torch.ones(c), requires_grad=True)

        # ── SUB-BLOCK 2: MLP ──────────────────────────────────────────────

        self.norm2   = nn.LayerNorm(c)
        mlp_hidden   = int(c * mlp_ratio)
        self.mlp_fc1 = nn.Linear(c, mlp_hidden)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(mlp_hidden, c)

        self.gamma2 = nn.Parameter(1e-2 * torch.ones(c), requires_grad=True)

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        """trunc_normal(std=0.02) — chuẩn Transformer/ViT init."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────
    def _valid_area(self, H: int, W: int) -> tuple[int, int]:
        """
        Tìm area hợp lệ lớn nhất ≤ self.area.
        Điều kiện: sqrt(area) nguyên và H, W chia hết cho sqrt(area).
        """
        for a in range(self.area, 0, -1):
            sq = int(math.isqrt(a))
            if sq * sq == a and H % sq == 0 and W % sq == 0:
                return a, sq
        return 1, 1

    # ──────────────────────────────────────────────────────────────────────
    def _split(self, feat: torch.Tensor, sq: int) -> torch.Tensor:
        """
        Chia feature map thành area vùng không gian (2D grid).
        (B, C, H, W) → (B*area, M, C)
        area = sq²,  M = (H//sq) × (W//sq)
        """
        B, C, H, W = feat.shape
        h, w = H // sq, W // sq
        feat = feat.view(B, C, sq, h, sq, w)
        feat = feat.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B,sq,sq,h,w,C)
        return feat.view(B * sq * sq, h * w, C)              # (B*area, M, C)

    # ──────────────────────────────────────────────────────────────────────
    def _merge(
        self,
        feat: torch.Tensor,
        B: int, H: int, W: int, sq: int,
    ) -> torch.Tensor:
        """
        Ghép các area về feature map không gian.
        (B*area, M, C) → (B, C, H, W)
        """
        h, w = H // sq, W // sq
        C    = feat.shape[-1]
        feat = feat.view(B, sq, sq, h, w, C)               # (B,sq,sq,h,w,C)
        feat = feat.permute(0, 5, 1, 3, 2, 4).contiguous() # (B,C,sq,h,sq,w)
        return feat.view(B, C, H, W)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, c, H, W)
        Returns:
            out: (B, c, H, W)
        """
        B, C, H, W   = x.shape
        area, sq     = self._valid_area(H, W)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 1 — 3-Stage Area Attention                       ║
        # ╚══════════════════════════════════════════════════════════════╝

        # Pre-norm: (B,C,H,W) → (B,H,W,C) → LN → (B,C,H,W)
        x_n = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # ── 4 Conv1×1 toàn cục (proj_d2 đã bỏ) ──────────────────────
        k1 = self.proj_k1(x_n)   # (B, K, H, W)
        k2 = self.proj_k2(x_n)   # (B, K, H, W)
        d1 = self.proj_d1(x_n)   # (B, D, H, W)
        zz = self.proj_z(x_n)    # (B, Z, H, W)

        # ── Chia area ─────────────────────────────────────────────────
        k1 = self._split(k1, sq)                      # (B*area, M, K)
        k2 = self._split(k2, sq).transpose(1, 2)      # (B*area, K, M)
        d1 = self._split(d1, sq)                      # (B*area, M, D)
        zz = self._split(zz, sq)                      # (B*area, M, Z)

        # ── Stage 1: K-space coarse attention — "where to look" ───────
        # L2-normalize → dot product ∈ [-1, 1] → loại overflow FP16
        k1_n = F.normalize(k1.float(), dim=-1, eps=1e-6).to(k1.dtype)
        k2_n = F.normalize(k2.float(), dim=-2, eps=1e-6).to(k2.dtype)
        out1 = torch.bmm(k1_n, k2_n)                 # (B*area, M, M) ∈[-1,1]

        # Safety net: triệt tiêu NaN/Inf nếu có
        out1 = torch.nan_to_num(out1, nan=0.0, posinf=1.0, neginf=-1.0)

        # Learnable temperature: clamp ≥ 0.1 để tránh attention quá peaked
        temp = self.temperature.clamp(min=0.1)
        out1 = F.softmax(out1 / temp, dim=-1)         # (B*area, M, M)

        # ── Stage 2: D-space feature aggregation — "what context" ─────
        hidden = torch.bmm(out1, d1)                  # (B*area, M, D)
        hidden = self.norm_hidden(hidden)

        # ── Stage 3: Gated Z-space value — "what to output" ───────────
        # SwiGLU: silu(Linear(D→Z)) × zz
        # · silu(·) = x·sigmoid(x): bounded below, smooth non-linearity
        # · hidden (D-space context) kiểm soát bao nhiêu Z-value đi qua
        gate = F.silu(self.dz_gate(hidden))           # (B*area, M, Z)
        out3 = gate * zz                               # (B*area, M, Z)
        out3 = self.norm_z(out3)

        # ── Ghép area → proj_out → Layer Scale → residual ─────────────
        out3 = self._merge(out3, B, H, W, sq)         # (B, Z, H, W)
        out3 = self.proj_out(out3)                    # (B, C, H, W)
        x = x + self.gamma1.view(1, C, 1, 1) * out3

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 2 — MLP (Feed-Forward Network)                   ║
        # ╚══════════════════════════════════════════════════════════════╝
        x_n = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
        x_n = self.norm2(x_n)
        x_n = self.mlp_fc2(self.mlp_act(self.mlp_fc1(x_n)))
        x_n = x_n.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        x = x + self.gamma2.view(1, C, 1, 1) * x_n

        return x

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"c={self.c}, K={self.K}, D={self.D}, Z={self.Z}, "
            f"area={self.area}, temp={self.temperature.item():.3f}"
        )
