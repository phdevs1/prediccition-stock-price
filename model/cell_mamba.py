# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .neural_operations import get_skip_connection
from .utils import get_stride_for_cell_type
from .bimamba_wrapper import BIMambaWrapper

class CellMamba(nn.Module):
    """
    Replacement Cell that uses BIMambaWrapper inside.
    API compatible with original Cell: forward(s) returns skip + 0.1*out
    """
    def __init__(self, Cin, Cout, cell_type, arch=None, use_se=False, vocab_in=None):
        super().__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, channel_mult=2)
        self.use_se = use_se
        self.Cin = Cin
        self.Cout = Cout
        self.bimamba = BIMambaWrapper(Cin, Cout, vocab_in=vocab_in)

    def forward(self, s):
        # skip branch (keeps behaviour)
        skip = self.skip(s)
        out = self.bimamba(s)
        # preserve the 0.1 scaling used in original Cell
        return skip + 0.1 * out
