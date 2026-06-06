# -*-Encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neural_operations import OPS, EncCombinerCell, DecCombinerCell, Conv2D, get_skip_connection, MambaOp
from .utils import get_stride_for_cell_type, get_arch_cells


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, channel_mult=2)
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            if i==0:
                primitive = arch[i]
                op = OPS[primitive](Cin, Cout, stride)
            else:
                primitive = arch[i]
                op = OPS[primitive](Cout, Cout, stride)
            self._ops.append(op)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)
        scale = 1.0 if all(isinstance(op, MambaOp) for op in self._ops) else 0.1
        return skip + scale * s



def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)


def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    # print(eps)
    z = eps.mul_(sigma).add_(mu)
    # print(z.shape)
    return z, eps


class Normal:
    def __init__(self, mu, log_sigma):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma)

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p


class NormalDecoder:
    def __init__(self, param):
        B, C, H, W = param.size()
        self.num_c = C // 2
        self.mu = param[:, :self.num_c, :, :]                                 # B, 3, H, W
        self.log_sigma = param[:, self.num_c:, :, :]                          # B, 3, H, W
        self.sigma = torch.exp(self.log_sigma) + 1e-2
        self.dist = Normal(self.mu, self.log_sigma)

    def log_prob(self, samples):
        return self.dist.log_p(samples)

    def sample(self,):
        x, _ = self.dist.sample()
        return x



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.channel_mult = args.channel_mult
        self.mult = args.mult
        self.prediction_length = args.prediction_length
        self.num_preprocess_blocks = args.num_preprocess_blocks
        self.num_preprocess_cells = args.num_preprocess_cells
        self.num_channels_enc = args.num_channels_enc
        # Si use_bimamba=True se usa el arch_type 'mamba_enc' que tiene MambaOp
        # en normal_enc y normal_dec en lugar de las ops convolucionales
        arch_type = 'mamba_enc' if getattr(args, 'use_bimamba', False) else 'res_mbconv'
        self.arch_instance = get_arch_cells(arch_type)
        self.stem = Conv2D(1, args.num_channels_enc, 3, padding=1, bias=True)
        self.num_latent_per_group = args.num_latent_per_group

        self.num_channels_dec = args.num_channels_dec
        self.groups_per_scale = args.groups_per_scale
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.input_size = args.embedding_dimension
        self.hidden_size = args.hidden_size
        self.projection = nn.Linear(args.embedding_dimension+args.hidden_size, args.target_dim)

        c_scaling = self.channel_mult ** (self.num_preprocess_blocks) #4
        spatial_scaling = 2 ** (self.num_preprocess_blocks) #4

        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), args.prediction_length// spatial_scaling,
                           (args.embedding_dimension + args.hidden_size + 1) // spatial_scaling)
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)

        self.pre_process = self.init_pre_process(args.mult)
        self.enc_tower = self.init_encoder_tower(self.mult)

        self.enc0 = nn.Sequential(nn.ELU(), Conv2D(self.num_channels_enc * self.mult,
                        self.num_channels_enc * self.mult, kernel_size=1, bias=True), nn.ELU())

        self.enc_sampler = self.init_sampler(self.mult)

        self.dec_tower = self.init_decoder_tower(self.mult)

        self.post_process = self.init_post_process(self.mult)
        self.image_conditional = nn.Sequential(nn.ELU(),
                             Conv2D(int(self.num_channels_dec * self.mult), 2, 3, padding=1, bias=True))

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(self.channel_mult * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch)
                    mult = self.channel_mult * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch)
                pre_process.append(cell)
        self.mult = mult
        return pre_process

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        for g in range(self.groups_per_scale):
            arch = self.arch_instance['normal_enc']
            num_c = int(self.num_channels_enc * mult)
            cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch)
            enc_tower.append(cell)

            if not (g == self.groups_per_scale - 1):
                num_ce = int(self.num_channels_enc * mult)
                num_cd = int(self.num_channels_dec * mult)
                cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                enc_tower.append(cell)

        self.mult = mult
        return enc_tower

    def init_decoder_tower(self, mult):
        dec_tower = nn.ModuleList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_dec * mult)
            if not (g == 0):
                arch = self.arch_instance['normal_dec']
                cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch)
                dec_tower.append(cell)
            cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
            dec_tower.append(cell)
        self.mult = mult
        return dec_tower

    def init_sampler(self, mult):
        enc_sampler = nn.ModuleList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_enc * mult)
            cell = Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
            enc_sampler.append(cell)
        mult = mult/self.channel_mult
        return enc_sampler

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / self.channel_mult)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch)
                    mult = mult / self.channel_mult
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch)
                post_process.append(cell)
        self.mult = mult
        return post_process

    def forward(self, x):
        s = self.stem(2 * x - 1.0)
        for cell in self.pre_process:
            s = cell(s)
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()
        idx_dec = 0
        ftr = self.enc0(s)   #conv
        param0 = self.enc_sampler[idx_dec](ftr) # another conv2d
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)
        z, _ = dist.sample()   #z_0
        s = self.prior_ftr0.unsqueeze(0) # random value
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        idx_dec = 0
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()    # z_n
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.post_process:
            s = cell(s)
        # print(s.shape)
        logits = self.image_conditional(s)
        logits = self.projection(logits[...,-(self.input_size + self.hidden_size):])
        return logits

    def decoder_output(self, logits):
        return NormalDecoder(logits)
