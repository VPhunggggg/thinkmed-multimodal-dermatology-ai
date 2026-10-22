"""
ProtoCohere — Proto-Coherence Module (PCM)
==========================================
2-stage object region identification trong feature space.

Stage 1 — Prototype Similarity:
  Tại mỗi vị trí (h,w), tính cosine similarity với K class prototypes.
  rough_map = max similarity qua tất cả class → vùng có thể là object.

Stage 2 — Coherence Expansion (k=3):
  Dùng 2 điều kiện để xác định vùng thuộc cùng object:
    a. Gần seed (max_pool k=3 trên rough_map)
    b. Nhất quán với neighborhood (cosine_sim(F, avg_pool(F, k=3)))
  refined_map = seed_spread × relu(coherence)

Output:
  diff   = F × refined_map          ← trực tiếp, không qua highlighted
  output = F + γ × diff             ← residual với Layer Scale γ=0.01

Prototype learning:
  · Giai đoạn đầu (epoch < warmup): fully learnable qua backprop
  · Giai đoạn sau (epoch ≥ warmup): thêm self-supervised EMA update
    → Positions vượt qua BOTH Stage 1 + Stage 2 (refined_map > threshold)
    → Feature tại vị trí đó → update prototype class tương ứng
    → Không cần external predictions, hoàn toàn self-contained

Author  : [Private]
License : [Private]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoCohere(nn.Module):
    """
    Proto-Coherence Module (PCM).

    Args:
        c1              (int)  : Input channels (= output, residual).
        c2              (int)  : Output channels (phải bằng c1).
        num_classes     (int)  : Số class. Mặc định: 5 (RTTS).
        warmup_epochs   (int)  : Epoch bắt đầu EMA update. Mặc định: 30.
        ema_momentum    (float): EMA decay. Mặc định: 0.99.
        ema_threshold   (float): Ngưỡng refined_map để chọn vị trí update. Mặc định: 0.15.

    YAML syntax (sau khi register):
        - [-1, 1, ProtoCohere, [512, 5]]
          #         c2    num_classes
    """

    def __init__(
        self,
        c1            : int,
        c2            : int,
        num_classes   : int   = 5,
        warmup_epochs : int   = 30,
        ema_momentum  : float = 0.99,
        ema_threshold : float = 0.15,
    ):
        super().__init__()
        assert c1 == c2, f"ProtoCohere là residual module — c1={c1} phải bằng c2={c2}."

        C = c1
        K = num_classes

        self.C             = C
        self.K             = K
        self.warmup_epochs = warmup_epochs
        self.ema_momentum  = ema_momentum
        self.ema_threshold = ema_threshold

        # Epoch counter — cập nhật từ training callback
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

        # ── Prototype Memory Bank ─────────────────────────────────────────
        # K unit vectors trong C-dim space
        # Học qua backprop từ detection loss (+ EMA sau warmup)
        self.prototypes = nn.Parameter(torch.empty(K, C))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        # ── Layer Scale ───────────────────────────────────────────────────
        # γ = 0.01: diff path nhỏ lúc đầu → training stable
        # Tự tăng khi model học được prototype có ý nghĩa
        self.gamma = nn.Parameter(torch.tensor(0.01))
        
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Orthogonality regularization — ngăn prototype collapse.

        Mục tiêu: các prototype VUÔNG GÓC với nhau trong C-dim space.
        proto[car] ⊥ proto[bus] ⊥ proto[bicycle] ...
        → Mỗi prototype encode đặc trưng RIÊNG của class mình

        Loss = ||P_norm @ P_norm.T - I||²_F
        · gram = I (identity) → prototypes hoàn toàn orthogonal → loss = 0
        · gram = ones         → tất cả collapse về 1 hướng → loss = lớn

        Chỉ tính trên off-diagonal để không phạt self-similarity
        (proto[k] · proto[k] = 1.0 là đúng, không cần phạt)
        """
        P    = F.normalize(self.prototypes, dim=-1)        # (K, C) unit vectors
        gram = P @ P.T                                      # (K, K) cosine similarity matrix
        I    = torch.eye(self.K, device=P.device, dtype=P.dtype)
        loss = (gram - I).pow(2)                           # penalty matrix (K, K)

        # Chỉ lấy off-diagonal (K*(K-1) phần tử)
        # Diagonal = self-similarity, luôn = 1.0 → không phạt
        mask        = ~torch.eye(self.K, dtype=torch.bool, device=P.device)
        off_diag    = loss[mask]

        return off_diag.mean()

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  STAGE 1 — Prototype Similarity                             ║
        # ║  "Vùng nào CÓ THỂ là object?"                               ║
        # ╚══════════════════════════════════════════════════════════════╝

        # Flatten spatial → sequence, normalize cho cosine sim
        x_flat = x.flatten(2).permute(0, 2, 1)                 # (B, N, C)
        x_norm = F.normalize(x_flat, dim=-1, eps=1e-6)         # (B, N, C) unit
        p_norm = F.normalize(self.prototypes, dim=-1, eps=1e-6) # (K, C)   unit

        # Cosine similarity: (B, N, K) ∈ [-1, 1]
        # sim[b,n,k] = "vị trí n của batch b gần prototype k đến đâu?"
        sim = x_norm @ p_norm.T                                  # (B, N, K)

        # Max similarity qua tất cả class → rough object probability
        # rough_map[b,n] = "vị trí n gần nhất prototype nào cũng được" ∈ [0,1]
        rough_map = sim.max(dim=-1).values                       # (B, N) ∈ [-1,1]
        rough_map = torch.sigmoid(rough_map)                     # → [0, 1]
        rough_map = rough_map.view(B, 1, H, W)                   # (B, 1, H, W)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  STAGE 2 — Coherence Expansion (k=3)                        ║
        # ║  "Vùng nào THUỘC CÙNG object?"                              ║
        # ╚══════════════════════════════════════════════════════════════╝

        # Điều kiện a: Gần seed → max_pool nở rộng 1 vị trí mỗi hướng
        # Mỗi seed kéo theo 8 neighbors trong 3×3
        seed_spread = F.max_pool2d(
            rough_map, kernel_size=3, stride=1, padding=1
        )                                                        # (B, 1, H, W)

        # Điều kiện b: Nhất quán với neighborhood
        # avg_pool(F, k=3) = trung bình 3×3 neighborhood của mỗi vị trí
        x_avg       = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)   # (B,C,H,W)
        x_dir       = F.normalize(x,     dim=1, eps=1e-6)       # unit vectors
        x_avg_dir   = F.normalize(x_avg, dim=1, eps=1e-6)
        # Cosine similarity along channel dim → scalar per position
        coherence   = (x_dir * x_avg_dir).sum(dim=1, keepdim=True)  # (B,1,H,W)∈[-1,1]
        # relu: chỉ giữ coherent positions (coherence > 0), loại bỏ outliers

        # Combined: vị trí được chấp nhận khi ĐẠT CẢ 2 ĐIỀU KIỆN
        refined_map = seed_spread * F.relu(coherence)            # (B, 1, H, W) ≥ 0

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  DIFF = F × refined_map  (trực tiếp, không qua highlighted) ║
        # ╚══════════════════════════════════════════════════════════════╝
        diff = x * refined_map                                   # (B, C, H, W)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  RESIDUAL OUTPUT                                            ║
        # ╚══════════════════════════════════════════════════════════════╝
        output = x + self.gamma * diff
        # = x × (1 + γ × refined_map)
        # γ=0.01 lúc đầu → output ≈ x (stable), tăng dần khi train

        # ── Self-supervised EMA prototype update (sau warmup) ─────────
        if self.training and self.current_epoch >= self.warmup_epochs:
            self._ema_update(x_flat, sim, refined_map)

        return output

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ema_update(
        self,
        x_flat     : torch.Tensor,    # (B, N, C) — features đã flatten
        sim        : torch.Tensor,    # (B, N, K) — similarity với prototypes
        refined_map: torch.Tensor,    # (B, 1, H, W) — refined object mask
    ) -> None:
        """
        Self-supervised EMA update prototype từ high-quality positions.

        Vị trí được chọn phải pass CẢ 2 bài test:
          1. rough_map cao (Stage 1): gần prototype → likely object
          2. coherence cao (Stage 2): nhất quán với neighbors → object interior

        Với mỗi class k: update prototype[k] bằng mean feature
        của các vị trí mà prototype[k] là best match VÀ refined_map > threshold.
        """
        B, N, C = x_flat.shape
        H_W_flat = refined_map.view(B, -1)                      # (B, N)

        # Class assignment: vị trí nào best match prototype nào?
        best_class = sim.max(dim=-1).indices                     # (B, N) ∈ [0, K-1]
        high_conf  = H_W_flat > self.ema_threshold               # (B, N) bool

        # Flatten batch
        feat_all  = x_flat.reshape(B * N, C)                    # (B*N, C)
        class_all = best_class.reshape(B * N)                    # (B*N,)
        conf_all  = high_conf.reshape(B * N)                     # (B*N,) bool

        for k in range(self.K):
            # Chọn vị trí: pass threshold VÀ class k là best match
            mask = conf_all & (class_all == k)                   # (B*N,) bool
            if mask.sum() < 1:
                continue

            # Mean feature tại các vị trí được chọn
            selected    = feat_all[mask]                         # (N_pos, C)
            batch_proto = F.normalize(selected.mean(0), dim=0)  # (C,) unit vector

            # EMA: prototype dần hội tụ về mean của high-quality class features
            self.prototypes.data[k] = (
                self.ema_momentum * self.prototypes.data[k]
                + (1.0 - self.ema_momentum) * batch_proto
            )

    # ──────────────────────────────────────────────────────────────────────
    def prototype_stats(self) -> dict:
        """
        Monitoring: đo độ đa dạng của prototypes.
        Gọi trong training callback để theo dõi collapse.

        Returns dict với:
          'mean_off_diag_sim': mean |cosine_sim| giữa các prototype ≠
                               < 0.2: healthy ✅
                               > 0.5: collapse risk ⚠️
          'gamma': giá trị Layer Scale hiện tại
        """
        with torch.no_grad():
            P    = F.normalize(self.prototypes, dim=-1)          # (K, C)
            gram = P @ P.T                                        # (K, K)
            K    = self.K
            mask = ~torch.eye(K, dtype=torch.bool, device=gram.device)
            div  = gram[mask].abs().mean().item()
        return {
            'mean_off_diag_sim': div,
            'gamma'            : self.gamma.item(),
            'ema_active'       : bool(self.current_epoch >= self.warmup_epochs),
        }

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"C={self.C}, K={self.K}, "
            f"warmup={self.warmup_epochs}, "
            f"gamma={self.gamma.item():.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════
# Training Callbacks
# ═════════════════════════════════════════════════════════════════════════

def make_protocohere_callbacks(
    log_every    : int   = 10,
    warmup_epochs: int   = 30,
    lambda_orth  : float = 0.01,   # ← thêm tham số này
):
    """
    lambda_orth: Hệ số orthogonality loss.
      0.01  → regularization nhẹ (khuyến nghị ban đầu)
      0.05  → mạnh hơn nếu diversity < 0.2 sau 50 epoch
      0.001 → rất nhẹ, chỉ hướng dẫn
    """

    # ── Callback 1: patch criterion để thêm orth loss ─────────────────
    def on_train_start(trainer):
        import types
        _model = trainer.model

        # Capture bound method GỐC trước khi patch
        # model.loss() là entry point: model(batch_dict) → self.loss(batch)
        _original_loss = _model.loss

        def _patched_loss(self, batch, preds=None):
            # Gọi loss gốc (box + cls + dfl)
            det_loss, det_items = _original_loss(batch, preds)

            # Cộng orthogonality loss từ tất cả ProtoCohere trong model
            orth = torch.tensor(0.0, device=det_loss.device)
            for m in _model.modules():
                if isinstance(m, ProtoCohere):
                    orth = orth + m.orthogonality_loss()

            total = det_loss + lambda_orth * orth
            return total, det_items

        # Gắn method mới vào model instance
        _model.loss = types.MethodType(_patched_loss, _model)
        print(
            f"[ProtoCohere] ✅ Orthogonality loss kích hoạt "
            f"(λ={lambda_orth})"
        )

    # ── Callback 2: cập nhật epoch counter ───────────────────────────
    def on_train_epoch_start(trainer):
        for m in trainer.model.modules():
            if isinstance(m, ProtoCohere):
                m.current_epoch.fill_(trainer.epoch)
        if trainer.epoch == warmup_epochs:
            print(f"\n[ProtoCohere] ✅ EMA update bắt đầu từ epoch {warmup_epochs}")

    # ── Callback 3: log diversity ────────────────────────────────────
    def on_train_epoch_end(trainer):
        if trainer.epoch % log_every != 0:
            return
        for name, m in trainer.model.named_modules():
            if not isinstance(m, ProtoCohere):
                continue
            stats = m.prototype_stats()
            div   = stats['mean_off_diag_sim']
            gamma = stats['gamma']
            ema   = '✅' if stats['ema_active'] else f'⏳(warmup {warmup_epochs})'

            if div > 0.5:
                status = "⚠️  COLLAPSE RISK — tăng lambda_orth"
            elif div > 0.3:
                status = "🟡 Monitor"
            else:
                status = "✅ Healthy"

            print(
                f"[PCM epoch {trainer.epoch}] {name}: "
                f"diversity={div:.3f} {status}  "
                f"gamma={gamma:.4f}  EMA={ema}"
            )

    return on_train_start, on_train_epoch_start, on_train_epoch_end
