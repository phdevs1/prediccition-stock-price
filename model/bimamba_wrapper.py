# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from samba.config.model_config import ModelArgs
# import Mamba lazily inside _lazy_init to avoid top-level import-time dependency errors

class BIMambaWrapper(nn.Module):
    """
    Wrapper that adapts NVAE Cell 4D tensors (B,C,H,W) to SAMBA Mamba (B,L,V).
    Lazy-instantiates Mamba when first forward is called (to know seq length H).
    """
    def __init__(self, Cin, Cout, vocab_in=None, mamba_d_model=64, mamba_n_layer=2, mamba_hid=128):
        super().__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.vocab_in = vocab_in
        self.mamba_d_model = mamba_d_model
        self.mamba_n_layer = mamba_n_layer
        self.mamba_hid = mamba_hid

        # lazy attributes
        self.mamba = None
        self.in_proj = None
        self.out_proj = None
        # normalization and FFN operate on flattened feature dim (N = C * W)
        self.ln1 = None
        self.ffn = None
        self.ln2 = None

    def _lazy_init(self, H, W):
        """Initialize projections and Mamba once H and W known."""
        N = self.Cin * W
        vocab_in = self.vocab_in or min(128, N)
        # build ModelArgs for Mamba
        try:
            from samba.models.mamba import Mamba
        except Exception as e:
            raise ImportError("Could not import samba.models.mamba.Mamba. Ensure samba dependencies (e.g. einops) are installed: {}".format(e))
        args = ModelArgs(d_model=self.mamba_d_model, n_layer=self.mamba_n_layer,
                         vocab_size=vocab_in, seq_in=H, seq_out=H)
        self.mamba = Mamba(args, self.mamba_hid)
        # projections
        self.in_proj = nn.Identity() if N == vocab_in else nn.Linear(N, vocab_in)
        self.out_proj = nn.Identity() if N == vocab_in else nn.Linear(vocab_in, N)
        # layernorm and ffn over N
        self.ln1 = nn.LayerNorm(N)
        self.ffn = nn.Sequential(nn.Linear(N, max(4, N // 2)), nn.SiLU(), nn.Linear(max(4, N // 2), N))
        self.ln2 = nn.LayerNorm(N)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        if self.mamba is None:
            self._lazy_init(H, W)
            # ensure parameters created in lazy init are on the same device as the input
            try:
                self.to(x.device)
            except Exception:
                pass

        # flatten to sequence: (B, L=H, N=C*W)
        seq = x.permute(0, 2, 3, 1).reshape(B, H, C * W)  # (B, L, N)
        # project to vocab_in if needed
        y = self.in_proj(seq)  # (B, L, V)
        # Mamba expects (B, L, V)
        y = self.mamba(y)      # (B, L, V)
        # back to N
        y = self.out_proj(y)   # (B, L, N)
        # residual + norms + FFN
        y = self.ln1(seq + y)
        y2 = self.ffn(y)
        y = self.ln2(y + y2)
        out = y.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return out

    def ensure_initialized(self, x_or_shape):
        """Public helper to ensure the internal Mamba is initialised.
        Accepts either a tensor with shape (B,C,H,W) or a tuple (H,W).
        """
        if isinstance(x_or_shape, tuple) or isinstance(x_or_shape, list):
            H, W = x_or_shape
        else:
            # assume tensor
            _, _, H, W = x_or_shape.shape
        if self.mamba is None:
            self._lazy_init(H, W)
        return
