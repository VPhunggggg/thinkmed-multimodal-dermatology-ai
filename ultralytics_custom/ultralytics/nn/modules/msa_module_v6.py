"""
MSABlock — Multi-Stage Attention Block
=======================================
Hỗ trợ 2 loại Position Bias cho ablation study:

  pos_bias='cartesian'  →  Cartesian RPB (Δrow, Δcol lookup table)
  pos_bias='polar'      →  Polar Relative Position Bias (PRPB)

─────────────────────────────────────────────────────────────────
Cartesian RPB (Swin-style):
  · Bảng bias learnable (2h-1)×(2w-1), tra theo index (Δrow, Δcol)
  · 1,521 params (area 20×20)
  · Proven trong Swin Transformer

Polar RPB (PRPB — ablation mới):
  · Với mỗi cặp token (A, B): tính r_rel, θ_rel
      r   = √(Δrow² + Δcol²) / r_max     ← normalized [0,1]
      θ   = atan2(Δcol, Δrow)
      enc = [sin(θ), cos(θ), r]           ← 3 features, không discontinuity
  · MLP nhỏ: Linear(3→16) → GELU → Linear(16→1) → scalar bias
  · 81 params (bất kể area size)
  · Translation equivariant ✅ (chỉ dùng Δrow, Δcol)
  · Encode cả distance lẫn direction tường minh
─────────────────────────────────────────────────────────────────

Author  : [Private]
License : [Private]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# Polar Relative Position Bias Module
# ═════════════════════════════════════════════════════════════════════

class PolarRelativeBias(nn.Module):
    """
    Polar Relative Position Bias (PRPB).

    Với mỗi cặp token (i, j) trong area (h×w):
      Δrow = row_j - row_i,   Δcol = col_j - col_i
      r    = √(Δrow²+Δcol²) / r_max    ← normalized ∈ [0,1]
      θ    = atan2(Δcol, Δrow)
      feat = [sin(θ), cos(θ), r]        ← 3D continuous encoding

    MLP: feat → bias scalar (M×M matrix)

    Tại self-token (i=i): r=0, sin_θ=0, cos_θ=1 → feat=[0,1,0] ổn định.

    Args:
        mlp_hidden (int): Hidden dim của MLP bias. Mặc định: 16.
    """

    def __init__(self, mlp_hidden: int = 16):
        super().__init__()

        # MLP nhỏ: (sin_θ, cos_θ, r) → scalar bias
        # Tổng: 3×16+16 + 16×1+1 = 81 params
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Cache polar features (CPU float32) theo (h, w)
        self._feat_cache: dict[tuple, torch.Tensor] = {}

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    def _build_polar_features(self, h: int, w: int) -> torch.Tensor:
        """
        Tính features polar cho tất cả cặp token trong area (h×w).
        Lưu trên CPU, dtype=float32.

        Returns:
            feat: (M, M, 3)  —  [sin_θ, cos_θ, r_norm]
        """
        M = h * w

        # Tọa độ (row, col) của M token
        rows = torch.arange(h).repeat_interleave(w)   # (M,)
        cols = torch.arange(w).repeat(h)               # (M,)

        # Relative displacement: (M, M)
        d_row = rows.unsqueeze(1) - rows.unsqueeze(0)  # (M, M): d_row[i,j] = row_j - row_i... wait
        # Actually: for pair (i→j):
        # Δrow = row_j - row_i, Δcol = col_j - col_i
        d_row = rows[None, :] - rows[:, None]           # (M, M): [i,j] = row_j - row_i
        d_col = cols[None, :] - cols[:, None]           # (M, M): [i,j] = col_j - col_i

        # Polar coordinates
        r = torch.sqrt(d_row.float()**2 + d_col.float()**2)   # (M, M)

        # Normalize r: r_max = √((h-1)²+(w-1)²), fallback 1 nếu area=1×1
        r_max = math.sqrt((h - 1)**2 + (w - 1)**2) if (h > 1 or w > 1) else 1.0
        r_norm = r / r_max                                      # (M, M) ∈ [0, 1]

        # Angle: atan2(Δcol, Δrow) — đơn vị radian
        # Self-token: atan2(0, 0) = 0 → sin=0, cos=1 → stable
        theta = torch.atan2(d_col.float(), d_row.float())       # (M, M) ∈ [-π, π]

        sin_t = torch.sin(theta)   # (M, M)
        cos_t = torch.cos(theta)   # (M, M)

        # Stack thành feature: (M, M, 3)
        feat = torch.stack([sin_t, cos_t, r_norm], dim=-1)

        return feat   # CPU, float32

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Tính và trả về bias matrix (M, M).

        Polar features được cache theo (h, w);
        MLP được apply mỗi forward (weights học được).
        """
        key = (h, w)
        if key not in self._feat_cache:
            self._feat_cache[key] = self._build_polar_features(h, w)

        # (M, M, 3) — move to device, cast to current dtype
        feat = self._feat_cache[key].to(device=device, dtype=dtype)

        # MLP: (M, M, 3) → (M, M, 1) → squeeze → (M, M)
        bias = self.mlp(feat).squeeze(-1)

        return bias   # (M, M)


# ═════════════════════════════════════════════════════════════════════
# MSABlock — unified với cả 2 loại bias
# ═════════════════════════════════════════════════════════════════════

class MSABlock(nn.Module):
    """
    Multi-Stage Attention Block (3-stage + configurable Position Bias).

    Sub-block 1: LN → 3-stage Area Attention + PosBias → LayerScale → residual
    Sub-block 2: LN → MLP (GELU)                       → LayerScale → residual

    Args:
        c          (int)  : Input/output channels.
        K          (int)  : Stage-1 key/query dim.  Mặc định: 64.
        D          (int)  : Stage-2 context dim.    Mặc định: 128.
        Z          (int)  : Stage-3 value dim.      Mặc định: c.
        area       (int)  : Số spatial area.        Mặc định: 4.
        mlp_ratio  (float): MLP expansion ratio.    Mặc định: 2.0.
        pos_bias   (str)  : 'cartesian' | 'polar' | 'none'.
                            Mặc định: 'cartesian'.
        max_area_h (int)  : Max area height cho Cartesian RPB. Mặc định: 20.
        max_area_w (int)  : Max area width  cho Cartesian RPB. Mặc định: 20.
        polar_mlp_hidden (int): Hidden dim của PRPB MLP. Mặc định: 16.
    """

    def __init__(
        self,
        c              : int,
        K              : int        = 64,
        D              : int        = 128,
        Z              : int | None = None,
        area           : int        = 4,
        mlp_ratio      : float      = 2.0,
        pos_bias       : str        = 'polar',
        max_area_h     : int        = 20,
        max_area_w     : int        = 20,
        polar_mlp_hidden: int       = 16,
    ):
        super().__init__()
        assert pos_bias in ('cartesian', 'polar', 'none'), \
            f"pos_bias phải là 'cartesian', 'polar' hoặc 'none', nhận: {pos_bias!r}"

        Z = Z if Z is not None else c

        self.c, self.K, self.D, self.Z = c, K, D, Z
        self.area     = area
        self.pos_bias = pos_bias
        self.max_area_h = max_area_h
        self.max_area_w = max_area_w

        # ── SUB-BLOCK 1: Area Attention ───────────────────────────────
        self.norm1    = nn.LayerNorm(c)

        self.proj_k1  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_k2  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_d1  = nn.Conv2d(c, D, 1, bias=False)
        self.proj_z   = nn.Conv2d(c, Z, 1, bias=False)
        self.proj_out = nn.Conv2d(Z, c, 1, bias=False)

        self.norm_hidden = nn.LayerNorm(D)
        self.norm_z      = nn.LayerNorm(Z)

        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.dz_gate     = nn.Linear(D, Z, bias=False)

        # ── Position Bias ─────────────────────────────────────────────
        if pos_bias == 'cartesian':
            n_bias = (2 * max_area_h - 1) * (2 * max_area_w - 1)
            self.rel_pos_bias_table = nn.Parameter(torch.zeros(n_bias))
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)
            self._cart_index_cache: dict[tuple, torch.Tensor] = {}

        elif pos_bias == 'polar':
            self.polar_bias = PolarRelativeBias(mlp_hidden=polar_mlp_hidden)

        # pos_bias == 'none': không có gì thêm

        # Layer Scale
        self.gamma1 = nn.Parameter(1e-2 * torch.ones(c))

        # ── SUB-BLOCK 2: MLP ──────────────────────────────────────────
        self.norm2   = nn.LayerNorm(c)
        mlp_h        = int(c * mlp_ratio)
        self.mlp_fc1 = nn.Linear(c, mlp_h)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(mlp_h, c)
        self.gamma2  = nn.Parameter(1e-2 * torch.ones(c))

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    def _valid_area(self, H: int, W: int) -> tuple[int, int]:
        for a in range(self.area, 0, -1):
            sq = int(math.isqrt(a))
            if sq * sq == a and H % sq == 0 and W % sq == 0:
                return a, sq
        return 1, 1

    # ──────────────────────────────────────────────────────────────────
    def _split(self, feat: torch.Tensor, sq: int) -> torch.Tensor:
        """(B, C, H, W) → (B*area, M, C)"""
        B, C, H, W = feat.shape
        h, w = H // sq, W // sq
        feat = feat.view(B, C, sq, h, sq, w)
        feat = feat.permute(0, 2, 4, 3, 5, 1).contiguous()
        return feat.view(B * sq * sq, h * w, C)

    # ──────────────────────────────────────────────────────────────────
    def _merge(self, feat: torch.Tensor, B: int, H: int, W: int, sq: int) -> torch.Tensor:
        """(B*area, M, C) → (B, C, H, W)"""
        h, w = H // sq, W // sq
        C    = feat.shape[-1]
        feat = feat.view(B, sq, sq, h, w, C)
        feat = feat.permute(0, 5, 1, 3, 2, 4).contiguous()
        return feat.view(B, C, H, W)

    # ──────────────────────────────────────────────────────────────────
    def _cartesian_bias(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Cartesian RPB: tra bảng (2*max_h-1)×(2*max_w-1) theo (Δrow, Δcol).
        Interpolate bicubic nếu area > max.
        """
        max_h, max_w = self.max_area_h, self.max_area_w
        key = (h, w)

        if h <= max_h and w <= max_w:
            # Tra trực tiếp
            if key not in self._cart_index_cache:
                coords_h = torch.arange(h)
                coords_w = torch.arange(w)
                gh, gw   = torch.meshgrid(coords_h, coords_w, indexing='ij')
                pos      = torch.stack([gh.flatten(), gw.flatten()])  # (2, M)
                delta    = pos[:, :, None] - pos[:, None, :]          # (2, M, M)
                delta    = delta.permute(1, 2, 0)                     # (M, M, 2)
                idx = (
                    (delta[..., 0] + max_h - 1) * (2 * max_w - 1)
                    + (delta[..., 1] + max_w - 1)
                )
                self._cart_index_cache[key] = idx
            idx  = self._cart_index_cache[key].to(device)
            bias = self.rel_pos_bias_table[idx.view(-1)].view(h * w, h * w)

        else:
            # Bicubic interpolation
            bias_2d = self.rel_pos_bias_table.view(
                1, 1, 2 * max_h - 1, 2 * max_w - 1
            ).float()
            bias_2d = F.interpolate(
                bias_2d, size=(2 * h - 1, 2 * w - 1),
                mode='bicubic', align_corners=False,
            ).squeeze().to(dtype)

            coords_h = torch.arange(h, device=device)
            coords_w = torch.arange(w, device=device)
            gh, gw   = torch.meshgrid(coords_h, coords_w, indexing='ij')
            pos      = torch.stack([gh.flatten(), gw.flatten()])
            delta    = pos[:, :, None] - pos[:, None, :]
            delta    = delta.permute(1, 2, 0)
            idx = (
                (delta[..., 0] + h - 1) * (2 * w - 1)
                + (delta[..., 1] + w - 1)
            )
            bias = bias_2d.view(-1)[idx.view(-1)].view(h * w, h * w)

        return bias   # (M, M)

    # ──────────────────────────────────────────────────────────────────
    def _get_pos_bias(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Dispatcher: trả về bias (M, M) hoặc None."""
        if self.pos_bias == 'cartesian':
            return self._cartesian_bias(h, w, device, dtype)
        elif self.pos_bias == 'polar':
            return self.polar_bias(h, w, device, dtype)
        return None   # 'none'

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, c, H, W)
        Returns:
            out: (B, c, H, W)
        """
        B, C, H, W   = x.shape
        area, sq     = self._valid_area(H, W)
        h_a, w_a     = H // sq, W // sq

        # ╔══════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 1 — 3-Stage Area Attention                   ║
        # ╚══════════════════════════════════════════════════════════╝
        x_n = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

        k1 = self.proj_k1(x_n)
        k2 = self.proj_k2(x_n)
        d1 = self.proj_d1(x_n)
        zz = self.proj_z(x_n)

        k1 = self._split(k1, sq)
        k2 = self._split(k2, sq).transpose(1, 2)
        d1 = self._split(d1, sq)
        zz = self._split(zz, sq)

        # Stage 1: L2-norm + temperature + position bias
        k1_n = F.normalize(k1.float(), dim=-1, eps=1e-6).to(k1.dtype)
        k2_n = F.normalize(k2.float(), dim=-2, eps=1e-6).to(k2.dtype)
        out1 = torch.bmm(k1_n, k2_n)
        out1 = torch.nan_to_num(out1, nan=0.0, posinf=1.0, neginf=-1.0)

        temp = self.temperature.clamp(min=0.1)
        out1 = out1 / temp

        # Cộng position bias (nếu có)
        bias = self._get_pos_bias(h_a, w_a, x.device, out1.dtype)
        if bias is not None:
            out1 = out1 + bias.unsqueeze(0)   # broadcast (B*area, M, M)

        out1 = F.softmax(out1, dim=-1)

        # Stage 2: D-space aggregation
        hidden = torch.bmm(out1, d1)
        hidden = self.norm_hidden(hidden)

        # Stage 3: SwiGLU gated Z-value
        gate = F.silu(self.dz_gate(hidden))
        out3 = gate * zz
        out3 = self.norm_z(out3)

        out3 = self._merge(out3, B, H, W, sq)
        out3 = self.proj_out(out3)
        x    = x + self.gamma1.view(1, C, 1, 1) * out3

        # ╔══════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 2 — MLP                                      ║
        # ╚══════════════════════════════════════════════════════════╝
        x_n = x.permute(0, 2, 3, 1)
        x_n = self.norm2(x_n)
        x_n = self.mlp_fc2(self.mlp_act(self.mlp_fc1(x_n)))
        x_n = x_n.permute(0, 3, 1, 2).contiguous()
        x   = x + self.gamma2.view(1, C, 1, 1) * x_n

        return x

    # ──────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"c={self.c}, K={self.K}, D={self.D}, Z={self.Z}, "
            f"area={self.area}, pos_bias={self.pos_bias!r}, "
            f"temp={self.temperature.item():.3f}"
        )
