"""
MGA — Memory Guided Attention (v2)
====================================
Scan degradation patterns trong hazy features → attention gate → residual.

Cải tiến so với v1:
  1. Depthwise Separable memory scanner: 8× nhẹ hơn Conv3×3 gốc
  2. Spatial + Channel attention thay bottleneck M→1:
       · Spatial gate (B,1,H,W): vị trí nào bị haze
       · Channel gate (B,C,1,1): channel nào bị ảnh hưởng
       · gate = spatial × channel → (B,C,H,W)
  3. Bỏ feature_processor (Conv1×1): gate đã đủ chọn lọc feature

Params tại P4 (C=512, M=64): ~39,937  vs  v1: ~558,147

Author  : [Private]
License : [Private]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGA(nn.Module):
    """
    Memory Guided Attention (v2).

    Args:
        c1           (int): Input channels.
        c2           (int): Output channels (phải bằng c1, residual).
        num_memories (int): Số memory activation channels. Mặc định: 64.

    YAML syntax:
        - [-1, 1, MGA, [512, 64]]   # P4: c2=512, M=64
        - [-1, 1, MGA, [256, 64]]   # P3: c2=256, M=64
    """

    def __init__(self, c1: int, c2: int, num_memories: int = 64):
        super().__init__()
        assert c1 == c2, (
            f"MGA là residual module — c1={c1} phải bằng c2={c2}."
        )

        C = c1
        M = num_memories

        # ── Memory Scanner: Depthwise Separable ──────────────────────────
        # DW Conv(C,C,3×3): detect local spatial patterns (haze texture)
        # PW Conv(C,M,1×1): cross-channel combination → M memory maps
        # Params: C×9 + C×M  vs  C×M×9 (gốc) → tiết kiệm ~8×
        self.mem_dw  = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.mem_pw  = nn.Conv2d(C, M, 1, bias=False)
        self.mem_bn  = nn.BatchNorm2d(M)
        self.mem_act = nn.ReLU(inplace=True)

        # ── Spatial Attention: "Vị trí nào có degradation?" ──────────────
        # M activation maps → 1 spatial weight map (B, 1, H, W)
        self.spatial_conv = nn.Conv2d(M, 1, 1, bias=False)
        self.spatial_bn   = nn.BatchNorm2d(1)

        # ── Channel Attention: "Channel nào bị ảnh hưởng bởi haze?" ──────
        # Squeeze: global avg pool M maps → vector
        # Excite : M → C/16 → C (SE-style)
        r = max(C // 16, 4)
        self.ch_pool = nn.AdaptiveAvgPool2d(1)         # (B, M, 1, 1)
        self.ch_fc1  = nn.Conv2d(M, r, 1, bias=False)  # (B, r, 1, 1)
        self.ch_act  = nn.ReLU(inplace=True)
        self.ch_fc2  = nn.Conv2d(r, C, 1, bias=False)  # (B, C, 1, 1)

        # ── Layer Scale ───────────────────────────────────────────────────
        # Khởi tạo 0.1: nhỏ lúc đầu → training stable
        # Tự tăng khi model học được attention có ý nghĩa
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

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # ── Memory Scanning ───────────────────────────────────────────────
        mem = self.mem_dw(x)                                        # (B, C, H, W)
        mem = self.mem_pw(mem)                                      # (B, M, H, W)
        mem = self.mem_act(self.mem_bn(mem))

        # L2-normalize theo channel dim → stability với AMP (float16)
        # cast float32 trước normalize để tránh eps≈0 trong float16
        mem = F.normalize(mem.float(), p=2, dim=1).to(x.dtype)     # (B, M, H, W)

        # ── Spatial Attention Gate ────────────────────────────────────────
        # "Vị trí (h,w) nào có haze/degradation pattern?"
        s = self.spatial_bn(self.spatial_conv(mem))                 # (B, 1, H, W)
        s = torch.clamp(s, min=-10, max=10)                         # prevent overflow
        s_gate = torch.sigmoid(s)                                   # (B, 1, H, W) ∈ [0,1]

        # ── Channel Attention Gate ────────────────────────────────────────
        # "Feature channel nào phản ứng với haze patterns?"
        c = self.ch_pool(mem)                                       # (B, M, 1, 1)
        c = self.ch_act(self.ch_fc1(c))                             # (B, r, 1, 1)
        c = self.ch_fc2(c)                                          # (B, C, 1, 1)
        c_gate = torch.sigmoid(c)                                   # (B, C, 1, 1) ∈ [0,1]

        # ── Combined Gate (CBAM-style) ────────────────────────────────────
        # Kết hợp không gian × channel → attention toàn diện
        gate = s_gate * c_gate                                      # (B, C, H, W)

        # ── Residual Output ───────────────────────────────────────────────
        # x × gate: highlight features tại degradation-aware regions
        # gamma=0.1: contribution được kiểm soát, tự điều chỉnh khi train
        return x + self.gamma * (x * gate)

    # ──────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        C = self.mem_dw.in_channels
        M = self.mem_pw.out_channels
        return f"C={C}, M={M}, gamma={self.gamma.item():.4f}"
    
    
class MGA_1(nn.Module):
    """
    Improved Memory Guided Attention Base
    
    Changes from original:
    1. Memory scanner: 5×5 conv with padding=2 (was 3×3 with manual pad)
    2. Feature processor: 3×3 conv with padding=1 (was 1×1)
    3. Direct memory → attention gate (removed fusion_conv)
    4. Channel-wise attention gate (was single channel)
    
    Args:
        channels: Number of input/output channels
        num_memories: Memory bank size (default: 64)
        kernel_size: Unused, kept for compatibility
    """
    
    def __init__(self, c1, c2, num_memories=64, kernel_size=3):
        super().__init__()
        channels = c1   # c2 = c1 (residual module)
        
        # Memory scanner: 5×5 conv for larger context
        self.memory_scanner = nn.Conv2d(
            channels,
            num_memories,
            kernel_size=5,       # Changed: 3→5 (larger receptive field)
            padding=2,           # Changed: 0→2 (auto padding, no manual F.pad)
            bias=False
        )
        
        # Attention gate generator: memory activations → channel attention
        # Maps num_memories to channels for element-wise multiplication
        self.attention_gate_conv = nn.Sequential(
            nn.Conv2d(num_memories, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Feature processor: 3×3 conv for spatial context
        self.feature_processor = nn.Sequential(
            nn.Conv2d(
                channels, 
                channels, 
                kernel_size=3,   # Changed: 1→3 (spatial processing)
                padding=1,       # Maintain spatial size
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable residual scale
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # 1. Scan for memory patterns (5×5 conv, no manual padding needed)
        memory_activations = self.memory_scanner(x)
        
        # 2. Normalize for stability
        memory_activations = F.normalize(memory_activations, p=2, dim=1)
        
        # 3. Generate channel-wise attention gate from memory
        attention_gate = self.attention_gate_conv(memory_activations)
        
        # 4. Clamp to prevent complete suppression or explosion
        attention_gate = torch.clamp(attention_gate, min=0.1, max=1.0)
        
        # 5. Process features with spatial context (3×3 conv)
        processed_features = self.feature_processor(x)
        
        # 6. Apply memory-guided attention (direct element-wise multiplication)
        gated_features = processed_features * attention_gate
        
        # 7. Residual connection with learnable scale
        return x + self.gate_scale * gated_features