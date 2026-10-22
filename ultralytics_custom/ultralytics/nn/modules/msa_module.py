"""
MSABlock — Multi-Stage Attention Block
=======================================
Hỗ trợ 2 loại Position Bias cho ablation study:

  pos_bias='cartesian'  →  Cartesian RPB (Δrow, Δcol lookup table)
  pos_bias='polar'      →  4-Corner Polar RPB (PRPB4C)

─────────────────────────────────────────────────────────────────
Cartesian RPB (Swin-style):
  · Bảng bias learnable (2h-1)×(2w-1), tra theo index (Δrow, Δcol)
  · 1,521 params (area 20×20)
  · Proven trong Swin Transformer

4-Corner Polar RPB (PRPB4C):
  · Với mỗi token, tính polar từ CẢ 4 GÓC của area (TL,TR,BL,BR):
      token_feat = [sin(θ_TL), cos(θ_TL), r_TL,
                    sin(θ_TR), cos(θ_TR), r_TR,
                    sin(θ_BL), cos(θ_BL), r_BL,
                    sin(θ_BR), cos(θ_BR), r_BR]  ← 12 features/token
  · Cặp (A,B): concat feat_A + feat_B → 24-dim
  · MLP: Linear(24→32) → GELU → Linear(32→1) → scalar bias
  · ~865 params, phân biệt vị trí token trong area
  · Symmetric với 4 góc → không favor top-left hay bottom-right
  · bias[i,j] ≠ bias[j,i]: directional asymmetry đúng thiết kế
─────────────────────────────────────────────────────────────────

Author  : [Private]
License : [Private]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# 4-Corner Polar Relative Position Bias
# ═════════════════════════════════════════════════════════════════════

class PolarRelativeBias4Corner(nn.Module):
    """
    4-Corner Polar Relative Position Bias (PRPB4C).

    Thay vì dùng relative displacement (Δrow, Δcol) làm reference duy nhất,
    module này tính polar coordinates của MỖI TOKEN từ CẢ 4 GÓC của area.

    Với token tại (row, col) trong area (h×w):
      Corners: TL=(0,0)  TR=(0,w-1)  BL=(h-1,0)  BR=(h-1,w-1)

      Từ mỗi góc c = (cr, cc):
        r_c     = √((row-cr)² + (col-cc)²) / r_max  ∈ [0,1]
        θ_c     = atan2(col-cc, row-cr)              ∈ [-π, π]
        feat_c  = [sin(θ_c), cos(θ_c), r_c]         ← 3 values

      token_feat = concat(feat_TL, feat_TR, feat_BL, feat_BR)  ← 12 values

    Với cặp (i, j):
      pair_feat = concat(token_feat[i], token_feat[j])          ← 24 values
      bias[i,j] = MLP(pair_feat)                               ← scalar

    Tính chất:
      · Symmetric w.r.t. 4 góc: encode đều từ mọi hướng
      · Phân biệt vị trí trong area: token ở TL ≠ token ở BR
      · bias[i,j] ≠ bias[j,i]: A nhìn B khác B nhìn A (đúng)
      · Token ở góc area có r nhỏ từ góc đó, lớn từ góc đối diện
        → MLP phân biệt được "token ở biên" vs "token ở trung tâm"

    Args:
        mlp_hidden (int): Hidden dim của bias MLP. Mặc định: 32.
    """

    def __init__(self, mlp_hidden: int = 32):
        super().__init__()

        # Input: 24 = 2 tokens × 4 corners × 3 features
        # Params: 24×32+32 + 32×1+1 = 801 params
        self.mlp = nn.Sequential(
            nn.Linear(24, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Cache token features (M, 12) trên CPU theo (h, w)
        # Tính 1 lần, dùng lại mọi forward pass
        self._tok_feat_cache: dict[tuple, torch.Tensor] = {}

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    def _build_token_features(self, h: int, w: int) -> torch.Tensor:
        """
        Tính 12-dim feature cho mỗi token trong area (h×w).
        Lưu trên CPU float32, reuse mọi forward pass.

        Returns:
            feat: (M, 12)  —  M = h×w
        """
        r_max = math.sqrt((h - 1)**2 + (w - 1)**2) if (h > 1 or w > 1) else 1.0

        # Tọa độ của M token trong area
        rows = torch.arange(h).repeat_interleave(w).float()  # (M,)
        cols = torch.arange(w).repeat(h).float()              # (M,)

        # 4 góc: TL, TR, BL, BR
        corners = [
            (0.,      0.   ),   # Top-Left
            (0.,      w-1. ),   # Top-Right
            (h-1.,    0.   ),   # Bottom-Left
            (h-1.,    w-1. ),   # Bottom-Right
        ]

        parts = []
        for cr, cc in corners:
            dr    = rows - cr                                  # (M,)
            dc    = cols - cc                                  # (M,)
            r     = torch.sqrt(dr**2 + dc**2) / r_max         # (M,) ∈ [0,1]
            theta = torch.atan2(dc, dr)                        # (M,) ∈ [-π,π]
            # sin/cos tránh discontinuity của góc
            parts.extend([torch.sin(theta), torch.cos(theta), r])

        # (M, 12): 4 corners × 3 features
        return torch.stack(parts, dim=-1)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        h     : int,
        w     : int,
        device: torch.device,
        dtype : torch.dtype,
    ) -> torch.Tensor:
        """
        Tính bias matrix (M, M) cho area (h×w).

        Token features (M, 12) được cache trên CPU.
        Pair features (M, M, 24) được compute trên GPU.

        Returns:
            bias: (M, M)
        """
        key = (h, w)
        if key not in self._tok_feat_cache:
            self._tok_feat_cache[key] = self._build_token_features(h, w)

        # Token features → GPU, current dtype
        tok = self._tok_feat_cache[key].to(device=device, dtype=dtype)  # (M, 12)
        M   = tok.shape[0]

        # Expand để tạo pair features:
        #   feat_i[i, j] = tok[i]  ← features của token i (Query)
        #   feat_j[i, j] = tok[j]  ← features của token j (Key)
        feat_i = tok.unsqueeze(1).expand(M, M, 12)   # (M, M, 12)
        feat_j = tok.unsqueeze(0).expand(M, M, 12)   # (M, M, 12)

        # Concat: (M, M, 24)
        # feat_i trước: "tôi đứng ở đây" → feat_j sau: "tôi nhìn về đó"
        pair = torch.cat([feat_i, feat_j], dim=-1)   # (M, M, 24)

        # MLP → scalar bias per pair
        bias = self.mlp(pair).squeeze(-1)             # (M, M)

        return bias


# ═════════════════════════════════════════════════════════════════════
# MSABlock — unified với cả 2 loại bias
# ═════════════════════════════════════════════════════════════════════

class MSABlock(nn.Module):
    """
    Multi-Stage Attention Block (3-stage + configurable Position Bias).

    Sub-block 1: LN → 3-stage Area Attention + PosBias → LayerScale → residual
    Sub-block 2: LN → MLP (GELU)                       → LayerScale → residual

    Args:
        c              (int)  : Input/output channels.
        K              (int)  : Stage-1 key/query dim.  Mặc định: 64.
        D              (int)  : Stage-2 context dim.    Mặc định: 128.
        Z              (int)  : Stage-3 value dim.      Mặc định: c.
        area           (int)  : Số spatial area.        Mặc định: 4.
        mlp_ratio      (float): MLP expansion ratio.    Mặc định: 2.0.
        pos_bias       (str)  : 'cartesian' | 'polar' | 'none'.
                                'polar' = 4-Corner PRPB4C.
                                Mặc định: 'cartesian'.
        max_area_h     (int)  : Max area height cho Cartesian RPB. Mặc định: 20.
        max_area_w     (int)  : Max area width  cho Cartesian RPB. Mặc định: 20.
        polar_mlp_hidden(int) : Hidden dim của PRPB4C MLP.        Mặc định: 32.
    """

    def __init__(
        self,
        c               : int,
        K               : int        = 64,
        D               : int        = 128,
        Z               : int | None = None,
        area            : int        = 4,
        mlp_ratio       : float      = 2.0,
        pos_bias        : str        = 'polar',
        max_area_h      : int        = 20,
        max_area_w      : int        = 20,
        polar_mlp_hidden: int        = 32,
    ):
        super().__init__()
        assert pos_bias in ('cartesian', 'polar', 'none'), \
            f"pos_bias phải là 'cartesian', 'polar' hoặc 'none', nhận: {pos_bias!r}"

        Z = Z if Z is not None else c

        self.c, self.K, self.D, self.Z = c, K, D, Z
        self.area       = area
        self.pos_bias   = pos_bias
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
            # 4-Corner PRPB4C thay thế PRPB gốc (3-dim relative)
            self.polar_bias = PolarRelativeBias4Corner(
                mlp_hidden=polar_mlp_hidden
            )

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
        """Cartesian RPB: tra bảng (2*max_h-1)×(2*max_w-1) theo (Δrow, Δcol)."""
        max_h, max_w = self.max_area_h, self.max_area_w
        key = (h, w)

        if h <= max_h and w <= max_w:
            if key not in self._cart_index_cache:
                coords_h = torch.arange(h)
                coords_w = torch.arange(w)
                gh, gw   = torch.meshgrid(coords_h, coords_w, indexing='ij')
                pos      = torch.stack([gh.flatten(), gw.flatten()])
                delta    = pos[:, :, None] - pos[:, None, :]
                delta    = delta.permute(1, 2, 0)
                idx = (
                    (delta[..., 0] + max_h - 1) * (2 * max_w - 1)
                    + (delta[..., 1] + max_w - 1)
                )
                self._cart_index_cache[key] = idx
            idx  = self._cart_index_cache[key].to(device)
            bias = self.rel_pos_bias_table[idx.view(-1)].view(h * w, h * w)
        else:
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

        return bias

    # ──────────────────────────────────────────────────────────────────
    def _get_pos_bias(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Dispatcher: trả về bias (M, M) hoặc None."""
        if self.pos_bias == 'cartesian':
            return self._cartesian_bias(h, w, device, dtype)
        elif self.pos_bias == 'polar':
            return self.polar_bias(h, w, device, dtype)
        return None

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
        # ║  SUB-BLOCK 1 — 3-Stage Area Attention + Position Bias   ║
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