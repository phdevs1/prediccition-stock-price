# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from .mamba_block import MambaBlock


@dataclass
class _MambaBlockArgs:
    d_model: int
    d_state: int = 16
    d_conv: int = 3
    expand: int = 2
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)


class BIMambaCell(nn.Module):
    """
    Bidirectional Mamba cell compatible with NVAE (B, C, H, W) tensors.

    Applies two MambaBlocks (forward + backward) along the H axis (temporal),
    treating each W position independently by folding W into the batch dimension.

    Input/output shape: (B, C, H, W)  — same as Conv2D-based Cell.
    """

    def __init__(self, channels: int, d_state: int = 8, d_conv: int = 3, expand: int = 1):
        super().__init__()
        args = _MambaBlockArgs(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.fwd = MambaBlock(args)
        self.bwd = MambaBlock(args)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) → (B*W, H, C): each spatial column becomes a sequence
        s = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        s = self.norm(s)
        y = self.fwd(s) + self.bwd(s.flip(1)).flip(1)
        # (B*W, H, C) → (B, C, H, W)
        return y.reshape(B, W, H, C).permute(0, 3, 2, 1)
