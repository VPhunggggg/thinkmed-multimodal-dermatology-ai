"""
ClassAwareMGA — Class-Aware Memory Guided Attention
=====================================================
Kết hợp MGA (degradation pattern detection) với Prototype (class identity).

Luồng xử lý 3 nhánh:

  Nhánh 1 — Class Identification:
    prototypes (K, C) → cosine_sim → softmax
    → class_weights (B, N, K): vị trí này thuộc class nào?

  Nhánh 2 — Memory Scanning (Depthwise Separable):
    DW + PW → M degradation patterns
    → mem (B, M, H, W): pattern haze nào xuất hiện ở đây?

  Routing — Class × Pattern:
    class_gate (K, M) learnable: "class k → dùng pattern nào?"
    → (B, N, K) @ (K, M) → weighted_mem (B, M, H, W)
    → "Car region → activate car-specific degradation memory"

  Spatial + Channel Attention → gate → residual

Tại sao đây là "memory" thực sự:
  · Prototypes = class memory (xe hơi trông như thế nào)
  · class_gate = degradation memory (xe hơi bị haze thì PATTERN nào quan trọng)
  · Routing = recall: "đây là car → nhớ lại car-specific haze pattern"

Params: ~58.9K (P4, C=512)  vs  MGA v2: ~56K  vs  ProtoCohere: ~2.5K

Author  : [Private]
License : [Private]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareMGA(nn.Module):
    """
    Class-Aware Memory Guided Attention.

    Args:
        c1          (int): Input channels (= output, residual).
        c2          (int): Output channels (phải bằng c1).
        num_classes (int): Số class. Mặc định: 5 (RTTS).
        num_memories(int): Số memory pattern. Mặc định: 64.

    YAML syntax:
        - [-1, 1, ClassAwareMGA, [512, 5, 64]]  # P4/P5
        - [-1, 1, ClassAwareMGA, [256, 5, 64]]  # P3
    """

    def __init__(
        self,
        c1          : int,
        c2          : int,
        num_classes : int = 5,
        num_memories: int = 64,
    ):
        super().__init__()
        assert c1 == c2, f"ClassAwareMGA là residual — c1={c1} phải bằng c2={c2}."

        C, K, M    = c1, num_classes, num_memories
        self.C     = C
        self.K     = K
        self.M     = M

        # ── Nhánh 1: Class Prototype Memory ──────────────────────────────
        # "Xe hơi/người/xe đạp trông như thế nào trong feature space"
        # Học hoàn toàn qua backprop + orthogonality loss
        self.prototypes = nn.Parameter(torch.empty(K, C))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        # ── Nhánh 2: Memory Scanner (Depthwise Separable) ─────────────────
        # DW: spatial haze texture patterns
        # PW: cross-channel combination → M memory activations
        self.mem_dw  = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.mem_pw  = nn.Conv2d(C, M, 1, bias=False)
        self.mem_bn  = nn.BatchNorm2d(M)
        self.mem_act = nn.ReLU(inplace=True)

        # ── Routing: Class → Pattern ──────────────────────────────────────
        # class_gate[k, m]: "class k quan tâm đến memory pattern m bao nhiêu?"
        # Learnable, sigmoid → [0,1]
        # Init 1/M: tất cả class bắt đầu coi mọi pattern ngang nhau
        self.class_gate = nn.Parameter(torch.ones(K, M) / M)

        # ── Spatial Attention ─────────────────────────────────────────────
        # Từ weighted_mem (B, M, H, W) → 1 spatial weight map
        self.spatial_conv = nn.Conv2d(M, 1, 1, bias=False)
        self.spatial_bn   = nn.BatchNorm2d(1)

        # ── Channel Attention ─────────────────────────────────────────────
        # Từ weighted_mem → C channel weights (SE-style)
        r            = max(C // 16, 4)
        self.ch_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1  = nn.Conv2d(M, r, 1, bias=False)
        self.ch_act  = nn.ReLU(inplace=True)
        self.ch_fc2  = nn.Conv2d(r, C, 1, bias=False)

        # ── Layer Scale ───────────────────────────────────────────────────
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Prototypes: trunc_normal (đã set trong __init__)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        K, M       = self.K, self.M
        N          = H * W

        # ╔══════════════════════════════════════════════════════════╗
        # ║  NHÁNH 1: Class Identification                          ║
        # ╚══════════════════════════════════════════════════════════╝
        # Flatten spatial → tính cosine sim với K prototypes
        x_flat  = x.flatten(2).permute(0, 2, 1)                   # (B, N, C)
        x_norm  = F.normalize(x_flat.float(), dim=-1, eps=1e-6).to(x.dtype)
        p_norm  = F.normalize(
            self.prototypes.float(), dim=-1, eps=1e-6
        ).to(x.dtype)                                              # (K, C)

        sim     = x_norm @ p_norm.T                                # (B, N, K)
        # softmax: mỗi vị trí có phân phối xác suất trên K classes
        weights = F.softmax(sim, dim=-1)                           # (B, N, K)
        # weights[b, n, k]: "vị trí n thuộc class k với xác suất bao nhiêu?"

        # ╔══════════════════════════════════════════════════════════╗
        # ║  NHÁNH 2: Memory Scanning                               ║
        # ╚══════════════════════════════════════════════════════════╝
        mem = self.mem_dw(x)                                       # (B, C, H, W)
        mem = self.mem_pw(mem)                                     # (B, M, H, W)
        mem = self.mem_act(self.mem_bn(mem))
        # L2-normalize trên channel dim → stability với AMP
        mem = F.normalize(mem.float(), p=2, dim=1).to(x.dtype)    # (B, M, H, W)
        # mem[b, m, h, w]: "pattern m có hiện diện tại vị trí (h,w) không?"

        # ╔══════════════════════════════════════════════════════════╗
        # ║  ROUTING: Class × Pattern → Weighted Memory             ║
        # ╚══════════════════════════════════════════════════════════╝
        # class_gate[k, m] ∈ [0,1]: "class k cần pattern m không?"
        gate = torch.sigmoid(self.class_gate)                      # (K, M)

        # routing[b, n, m] = sum_k(weights[b,n,k] × gate[k,m])
        # = "tại vị trí n của batch b, cần dùng pattern m bao nhiêu?"
        routing  = weights @ gate                                  # (B, N, M)
        routing  = routing.permute(0, 2, 1).view(B, M, H, W)      # (B, M, H, W)

        # Áp routing lên memory activations
        w_mem = mem * routing                                      # (B, M, H, W)
        # w_mem: memory patterns đã được filter bởi class identity
        # "Car region → car-specific patterns sẽ mạnh hơn"

        # ╔══════════════════════════════════════════════════════════╗
        # ║  SPATIAL + CHANNEL ATTENTION                            ║
        # ╚══════════════════════════════════════════════════════════╝
        # Spatial: vị trí nào cần được attend
        s      = self.spatial_bn(self.spatial_conv(w_mem))        # (B, 1, H, W)
        s      = torch.clamp(s, -10, 10)
        s_gate = torch.sigmoid(s)                                  # (B, 1, H, W)

        # Channel: channel nào bị ảnh hưởng bởi class-specific degradation
        c      = self.ch_pool(w_mem)                               # (B, M, 1, 1)
        c      = self.ch_act(self.ch_fc1(c))                       # (B, r, 1, 1)
        c      = self.ch_fc2(c)                                    # (B, C, 1, 1)
        c_gate = torch.sigmoid(c)                                  # (B, C, 1, 1)

        # Combined gate
        gate_final = s_gate * c_gate                               # (B, C, H, W)

        # ╔══════════════════════════════════════════════════════════╗
        # ║  RESIDUAL OUTPUT                                        ║
        # ╚══════════════════════════════════════════════════════════╝
        return x + self.gamma * (x * gate_final)

    # ──────────────────────────────────────────────────────────────────────
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Ép prototypes vuông góc → mỗi class có đặc trưng riêng biệt.
        Chỉ phạt off-diagonal (self-similarity luôn =1, không phạt).
        """
        P    = F.normalize(self.prototypes, dim=-1)                # (K, C)
        gram = P @ P.T                                              # (K, K)
        I    = torch.eye(self.K, device=P.device, dtype=P.dtype)
        diff = (gram - I).pow(2)
        mask = ~torch.eye(self.K, dtype=torch.bool, device=P.device)
        return diff[mask].mean()

    # ──────────────────────────────────────────────────────────────────────
    def module_stats(self) -> dict:
        """
        Monitor 2 chỉ số diversity:
          proto_diversity: prototypes khác nhau đến đâu?
          gate_diversity : routing pattern của các class khác nhau đến đâu?
        Cả 2 < 0.3 là healthy ✅.
        """
        with torch.no_grad():
            # Prototype diversity
            P      = F.normalize(self.prototypes, dim=-1)
            gram_p = P @ P.T
            mask   = ~torch.eye(self.K, dtype=torch.bool, device=gram_p.device)
            p_div  = gram_p[mask].abs().mean().item()

            # Gate diversity: các class có routing pattern khác nhau không?
            G      = F.normalize(self.class_gate, dim=-1)
            gram_g = G @ G.T
            g_div  = gram_g[mask].abs().mean().item()

        return {
            'proto_diversity': p_div,
            'gate_diversity' : g_div,
            'gamma'          : self.gamma.item(),
        }

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"C={self.C}, K={self.K}, M={self.M}, "
            f"gamma={self.gamma.item():.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════
# Training Callbacks
# ═════════════════════════════════════════════════════════════════════════

def make_class_aware_mga_callbacks(
    lambda_orth: float = 0.01,
    log_every  : int   = 10,
):
    """
    Tạo 2 callbacks cho ClassAwareMGA:
      1. on_train_start: patch model.loss() để thêm orthogonality loss
      2. on_train_epoch_end: monitor prototype + gate diversity

    Cách dùng trong code_train.py:
        from class_aware_mga import make_class_aware_mga_callbacks
        cb_start, cb_end = make_class_aware_mga_callbacks(lambda_orth=0.01)
        model.add_callback('on_train_start',     cb_start)
        model.add_callback('on_train_epoch_end', cb_end)
    """

    # ── Callback 1: Patch model.loss() ────────────────────────────────────
    def on_train_start(trainer):
        import types
        _model         = trainer.model
        _original_loss = _model.loss

        def _patched_loss(self, batch, preds=None):
            det_loss, det_items = _original_loss(batch, preds)

            # Orthogonality loss từ tất cả ClassAwareMGA instances
            orth = torch.tensor(0.0, device=det_loss.device)
            for m in _model.modules():
                if isinstance(m, ClassAwareMGA):
                    orth = orth + m.orthogonality_loss()

            total = det_loss + lambda_orth * orth
            return total, det_items

        _model.loss = types.MethodType(_patched_loss, _model)
        print(
            f"[ClassAwareMGA] ✅ Orthogonality loss kích hoạt "
            f"(λ={lambda_orth})"
        )

    # ── Callback 2: Monitor diversity ─────────────────────────────────────
    def on_train_epoch_end(trainer):
        if trainer.epoch % log_every != 0:
            return

        for name, m in trainer.model.named_modules():
            if not isinstance(m, ClassAwareMGA):
                continue

            stats  = m.module_stats()
            p_div  = stats['proto_diversity']
            g_div  = stats['gate_diversity']
            gamma  = stats['gamma']

            # Prototype status
            p_status = (
                "⚠️  COLLAPSE" if p_div > 0.5 else
                "🟡 Monitor"  if p_div > 0.3 else
                "✅ OK"
            )
            # Gate status
            g_status = (
                "⚠️  SAME ROUTING" if g_div < 0.05 else
                "🟡 Monitor"       if g_div < 0.15 else
                "✅ OK"
            )

            print(
                f"[CAMGA epoch {trainer.epoch}] {name}: "
                f"proto_div={p_div:.3f}{p_status}  "
                f"gate_div={g_div:.3f}{g_status}  "
                f"gamma={gamma:.4f}"
            )

    return on_train_start, on_train_epoch_end