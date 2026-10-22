"""
MSABlock — Multi-Stage Attention Block  (+ Relative Position Bias)
===================================================================

Thay đổi so với phiên bản 3-stage:
  + Relative Position Bias (RPB) thêm vào attention logits trước softmax.

Cách hoạt động của RPB:
  · Với area kích thước (h×w), mỗi cặp token (i1,j1)↔(i2,j2) có
    relative position (Δrow, Δcol) = (i1-i2, j1-j2).
  · Δrow ∈ [-(h-1), h-1],  Δcol ∈ [-(w-1), w-1]
    → tổng cộng (2h-1)×(2w-1) cặp tương đối.
  · Lưu một bảng bias learnable kích thước (2*max_h-1)×(2*max_w-1).
  · Với area bất kỳ ≤ max: tra trực tiếp từ bảng.
  · Với area > max: nội suy bicubic bảng bias rồi tra.

Tính chất:
  · Translation equivariant: 2 patch giống nhau ở 2 vị trí khác nhau
    trong area cho cùng attention pattern → đúng với detection.
  · Không phụ thuộc vào vị trí tuyệt đối → không bị lệch khi thay đổi
    input resolution.
  · Chỉ thêm (2*max_h-1)*(2*max_w-1) params — rất nhỏ.

Author  : [Private]
License : [Private]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSABlock(nn.Module):
    """
    Multi-Stage Attention Block (3-stage + Relative Position Bias).

    Sub-block 1: LN → 3-stage Area Attention + RPB → Layer Scale → residual
    Sub-block 2: LN → MLP (GELU)                  → Layer Scale → residual

    3-stage attention trong mỗi area (h×w):
      Stage 1: L2-norm(K1) · L2-norm(K2)ᵀ + RPB → softmax  →  "where"
      Stage 2: attention × D1 → LN                          →  "context"
      Stage 3: SwiGLU(Linear(D→Z)) × Z → LN                →  "what"

    Args:
        c          (int)  : Input/output channels.
        K          (int)  : Stage-1 key/query dim. Mặc định: 64.
        D          (int)  : Stage-2 context dim.   Mặc định: 128.
        Z          (int)  : Stage-3 value dim.     Mặc định: c.
        area       (int)  : Số spatial area.       Mặc định: 4.
        mlp_ratio  (float): MLP expansion ratio.   Mặc định: 2.0.
        max_area_h (int)  : Chiều cao max của 1 area để init bảng bias.
                            Mặc định: 20  (= 40×40 / sqrt(4) tại P4-640px).
        max_area_w (int)  : Chiều rộng max của 1 area. Mặc định: 20.
    """

    def __init__(
        self,
        c          : int,
        K          : int        = 64,
        D          : int        = 128,
        Z          : int | None = None,
        area       : int        = 4,
        mlp_ratio  : float      = 2.0,
        max_area_h : int        = 20,
        max_area_w : int        = 20,
    ):
        super().__init__()
        Z = Z if Z is not None else c

        self.c, self.K, self.D, self.Z = c, K, D, Z
        self.area = area
        self.max_area_h = max_area_h
        self.max_area_w = max_area_w

        # ── SUB-BLOCK 1: Area Attention ───────────────────────────────────

        self.norm1    = nn.LayerNorm(c)

        # 4 Conv1×1 toàn cục (proj_d2 đã bỏ từ 3-stage redesign)
        self.proj_k1  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_k2  = nn.Conv2d(c, K, 1, bias=False)
        self.proj_d1  = nn.Conv2d(c, D, 1, bias=False)
        self.proj_z   = nn.Conv2d(c, Z, 1, bias=False)
        self.proj_out = nn.Conv2d(Z, c, 1, bias=False)

        self.norm_hidden = nn.LayerNorm(D)
        self.norm_z      = nn.LayerNorm(Z)

        # Learnable temperature — khởi tạo 0.5, clamp ≥ 0.1
        self.temperature = nn.Parameter(torch.tensor(0.5))

        # SwiGLU gate: Linear(D→Z)
        self.dz_gate = nn.Linear(D, Z, bias=False)

        # ── Relative Position Bias ────────────────────────────────────────
        # Bảng bias shape: (2*max_h-1) × (2*max_w-1)
        # Đại diện cho tất cả cặp (Δrow, Δcol) trong area lớn nhất.
        n_bias = (2 * max_area_h - 1) * (2 * max_area_w - 1)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(n_bias))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Cache index CPU (không serialize — recompute khi cần)
        self._rel_index_cache: dict[tuple, torch.Tensor] = {}

        # Layer Scale
        self.gamma1 = nn.Parameter(1e-2 * torch.ones(c), requires_grad=True)

        # ── SUB-BLOCK 2: MLP ──────────────────────────────────────────────

        self.norm2   = nn.LayerNorm(c)
        mlp_hidden   = int(c * mlp_ratio)
        self.mlp_fc1 = nn.Linear(c, mlp_hidden)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(mlp_hidden, c)
        self.gamma2  = nn.Parameter(1e-2 * torch.ones(c), requires_grad=True)

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self):
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
        for a in range(self.area, 0, -1):
            sq = int(math.isqrt(a))
            if sq * sq == a and H % sq == 0 and W % sq == 0:
                return a, sq
        return 1, 1

    # ──────────────────────────────────────────────────────────────────────
    def _split(self, feat: torch.Tensor, sq: int) -> torch.Tensor:
        """(B, C, H, W) → (B*area, M, C)"""
        B, C, H, W = feat.shape
        h, w = H // sq, W // sq
        feat = feat.view(B, C, sq, h, sq, w)
        feat = feat.permute(0, 2, 4, 3, 5, 1).contiguous()
        return feat.view(B * sq * sq, h * w, C)

    # ──────────────────────────────────────────────────────────────────────
    def _merge(self, feat: torch.Tensor, B: int, H: int, W: int, sq: int) -> torch.Tensor:
        """(B*area, M, C) → (B, C, H, W)"""
        h, w = H // sq, W // sq
        C    = feat.shape[-1]
        feat = feat.view(B, sq, sq, h, w, C)
        feat = feat.permute(0, 5, 1, 3, 2, 4).contiguous()
        return feat.view(B, C, H, W)

    # ──────────────────────────────────────────────────────────────────────
    def _build_rel_index(self, h: int, w: int) -> torch.Tensor:
        """
        Xây dựng index (M, M) để tra bảng bias.
        Mỗi entry [i, j] = vị trí trong bảng (2*max_h-1)×(2*max_w-1)
        tương ứng với relative position của token i so với token j.

        Lưu trên CPU; chuyển sang device trong forward.
        """
        max_h, max_w = self.max_area_h, self.max_area_w

        # Tạo meshgrid vị trí trong area
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        pos = torch.stack([grid_h.flatten(), grid_w.flatten()])   # (2, M)

        # Relative position: Δrow, Δcol ∈ [-(size-1), size-1]
        delta = pos[:, :, None] - pos[:, None, :]  # (2, M, M)
        delta = delta.permute(1, 2, 0)             # (M, M, 2): [...,0]=Δrow, [...,1]=Δcol

        # Map sang 1D index của bảng (2*max_h-1) × (2*max_w-1)
        # Δrow + (max_h-1) ∈ [max_h-h, max_h+h-2] ⊂ [0, 2*max_h-2]
        idx = (
            (delta[..., 0] + max_h - 1) * (2 * max_w - 1)
            + (delta[..., 1] + max_w - 1)
        )  # (M, M)

        return idx   # CPU tensor, dtype=long

    # ──────────────────────────────────────────────────────────────────────
    def _get_rel_pos_bias(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Trả về bias matrix (M, M) cho area kích thước h×w.

        · h ≤ max_area_h và w ≤ max_area_w: tra trực tiếp bảng bias.
        · Ngược lại: nội suy bicubic bảng bias lên kích thước cần thiết.
        """
        max_h, max_w = self.max_area_h, self.max_area_w
        cache_key = (h, w)

        # ── Trường hợp 1: area ≤ max → tra trực tiếp ─────────────────────
        if h <= max_h and w <= max_w:
            if cache_key not in self._rel_index_cache:
                self._rel_index_cache[cache_key] = self._build_rel_index(h, w)

            idx  = self._rel_index_cache[cache_key].to(device)   # (M, M)
            bias = self.rel_pos_bias_table[idx.view(-1)].view(h * w, h * w)

        # ── Trường hợp 2: area > max → nội suy bảng bias ─────────────────
        else:
            # Reshape bảng thành 2D: (2*max_h-1, 2*max_w-1)
            bias_2d = self.rel_pos_bias_table.view(
                1, 1, 2 * max_h - 1, 2 * max_w - 1
            ).float()

            # Nội suy bicubic lên (2h-1, 2w-1)
            bias_2d = F.interpolate(
                bias_2d,
                size=(2 * h - 1, 2 * w - 1),
                mode='bicubic',
                align_corners=False,
            ).squeeze().to(dtype)    # (2h-1, 2w-1)

            # Xây index cho bảng đã nội suy
            # → max_h và max_w đặt = h, w để _build_rel_index đúng
            coords_h = torch.arange(h, device=device)
            coords_w = torch.arange(w, device=device)
            grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
            pos = torch.stack([grid_h.flatten(), grid_w.flatten()])
            delta = pos[:, :, None] - pos[:, None, :]          # (2, M, M)
            delta = delta.permute(1, 2, 0)                     # (M, M, 2)
            idx = (
                (delta[..., 0] + h - 1) * (2 * w - 1)
                + (delta[..., 1] + w - 1)
            )   # (M, M) — index vào bảng (2h-1)×(2w-1)

            bias = bias_2d.view(-1)[idx.view(-1)].view(h * w, h * w)

        return bias   # (M, M)

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
        h_area = H // sq
        w_area = W // sq

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 1 — 3-Stage Area Attention + RPB                 ║
        # ╚══════════════════════════════════════════════════════════════╝

        # Pre-norm
        x_n = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # 4 Conv1×1 toàn cục
        k1 = self.proj_k1(x_n)   # (B, K, H, W)
        k2 = self.proj_k2(x_n)   # (B, K, H, W)
        d1 = self.proj_d1(x_n)   # (B, D, H, W)
        zz = self.proj_z(x_n)    # (B, Z, H, W)

        # Chia area
        k1 = self._split(k1, sq)                       # (B*area, M, K)
        k2 = self._split(k2, sq).transpose(1, 2)       # (B*area, K, M)
        d1 = self._split(d1, sq)                       # (B*area, M, D)
        zz = self._split(zz, sq)                       # (B*area, M, Z)

        # ── Stage 1: K-space coarse attention + RPB ───────────────────
        k1_n = F.normalize(k1.float(), dim=-1, eps=1e-6).to(k1.dtype)
        k2_n = F.normalize(k2.float(), dim=-2, eps=1e-6).to(k2.dtype)
        out1 = torch.bmm(k1_n, k2_n)                  # (B*area, M, M)
        out1 = torch.nan_to_num(out1, nan=0.0, posinf=1.0, neginf=-1.0)

        # Chia nhiệt độ rồi cộng RPB — đúng thứ tự (bias không scale theo temp)
        temp = self.temperature.clamp(min=0.1)
        out1 = out1 / temp                             # scale similarity

        # RPB: (M, M) → broadcast lên (B*area, M, M)
        rpb  = self._get_rel_pos_bias(h_area, w_area, x.device, out1.dtype)
        out1 = out1 + rpb.unsqueeze(0)                 # (B*area, M, M)

        out1 = F.softmax(out1, dim=-1)

        # ── Stage 2: D-space feature aggregation ──────────────────────
        hidden = torch.bmm(out1, d1)                   # (B*area, M, D)
        hidden = self.norm_hidden(hidden)

        # ── Stage 3: Gated Z-space value (SwiGLU) ─────────────────────
        gate = F.silu(self.dz_gate(hidden))            # (B*area, M, Z)
        out3 = gate * zz                               # (B*area, M, Z)
        out3 = self.norm_z(out3)

        # Ghép area → proj_out → Layer Scale → residual
        out3 = self._merge(out3, B, H, W, sq)          # (B, Z, H, W)
        out3 = self.proj_out(out3)                     # (B, C, H, W)
        x    = x + self.gamma1.view(1, C, 1, 1) * out3

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  SUB-BLOCK 2 — MLP                                          ║
        # ╚══════════════════════════════════════════════════════════════╝

        x_n = x.permute(0, 2, 3, 1)
        x_n = self.norm2(x_n)
        x_n = self.mlp_fc2(self.mlp_act(self.mlp_fc1(x_n)))
        x_n = x_n.permute(0, 3, 1, 2).contiguous()
        x   = x + self.gamma2.view(1, C, 1, 1) * x_n

        return x

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"c={self.c}, K={self.K}, D={self.D}, Z={self.Z}, "
            f"area={self.area}, temp={self.temperature.item():.3f}, "
            f"max_area=({self.max_area_h},{self.max_area_w})"
        )
