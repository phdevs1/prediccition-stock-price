# -*- coding: utf-8 -*-
"""
Mamba model (local copy for samba)
"""
import torch
import torch.nn as nn
from .normalization import RMSNorm
from .mamba_block import MambaBlock


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x))
        return output


class Mamba(nn.Module):
    def __init__(self, args, hid):
        super().__init__()
        self.args = args
        self.nl = args.n_layer
        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.lin = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            )
        ] + [
            nn.Sequential(
                RMSNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            ) for _ in range(args.n_layer - 2)
        ] + [
            nn.Sequential(
                RMSNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            )
        ])
        self.norm_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size)
        self.proj = nn.Sequential(nn.Linear(args.seq_in, hid), nn.ReLU(), nn.Linear(hid, args.seq_in))

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x1 = x; x2 = x
        for i in range(self.nl):
            x1 = self.layers[i](x1)
            x2 = self.layers2[i](x2.flip([1]))
            x = x1 + x2.flip([1]) + x
            x = self.lin[i](x.permute(0, 2, 1)).permute(0, 2, 1) + x
            x1 = x; x2 = x
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
