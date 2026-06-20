# -*-Encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .neural_operations import (
    OPS,
    EncCombinerCell,
    DecCombinerCell,
    Conv2D,
    get_skip_connection,
)
from .utils import (
    get_stride_for_cell_type,
    get_arch_cells,
)
from samba.models import BIMambaCell


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, channel_mult=2)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            if i == 0:
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
        return skip + 0.1 * s


class CellMamba(nn.Module):
    """Drop-in replacement for normal (stride=1) Cell using BI-Mamba.

    Only valid for cell types where Cin == Cout (normal_pre, normal_enc,
    normal_dec, normal_post). Down/up cells must remain as Cell.
    """

    def __init__(self, C, cell_type, d_state: int = 8, expand: int = 1):
        super(CellMamba, self).__init__()
        self.cell_type = cell_type
        self.bi_mamba = BIMambaCell(C, d_state=d_state, expand=expand)

    def forward(self, s):
        # Gradient checkpointing: recompute activations in backward instead of
        # storing them, trading compute for memory on the large B*W effective batch.
        return s + 0.1 * checkpoint(self.bi_mamba, s, use_reentrant=False)


def soft_clamp5(x: torch.Tensor):
    return x.div(5.0).tanh_().mul(5.0)


def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class Normal:
    def __init__(self, mu, log_sigma, temp=1.0):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma)
        if temp != 1.0:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = (
            -0.5 * normalized_samples * normalized_samples
            - 0.5 * np.log(2 * np.pi)
            - torch.log(self.sigma)
        )
        return log_p

    def kl(self, prior):
        """KL(self || prior) por elemento, entre dos Normales (estilo NVAE)."""
        term1 = (self.mu - prior.mu) / prior.sigma
        term2 = self.sigma / prior.sigma
        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)


class NormalDecoder:
    def __init__(self, param):
        B, C, H, W = param.size()
        self.num_c = C // 2
        self.mu = param[:, : self.num_c, :, :]  # B, 3, H, W
        self.log_sigma = param[:, self.num_c :, :, :]  # B, 3, H, W
        self.sigma = torch.exp(self.log_sigma) + 1e-2
        self.dist = Normal(self.mu, self.log_sigma)

    def log_prob(self, samples):
        return self.dist.log_p(samples)

    def sample(
        self,
    ):
        x, _ = self.dist.sample()
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        # KL latente acumulada en el ultimo forward (por-batch). Se lee desde el
        # bucle de entrenamiento sin alterar la firma de forward.
        self.kl_loss = None

        self.channel_mult = args.channel_mult
        self.mult = args.mult
        self.prediction_length = args.prediction_length
        self.num_preprocess_blocks = args.num_preprocess_blocks
        self.num_preprocess_cells = args.num_preprocess_cells
        self.num_channels_enc = args.num_channels_enc
        self.arch_instance = get_arch_cells(args.arch_instance)
        self.stem = Conv2D(1, args.num_channels_enc, 3, padding=1, bias=True)
        self.num_latent_per_group = args.num_latent_per_group

        self.num_channels_dec = args.num_channels_dec
        self.groups_per_scale = args.groups_per_scale
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.use_se = False
        self.use_bimamba = getattr(args, "use_bimamba", False)
        self.bimamba_d_state = getattr(args, "bimamba_d_state", 8)
        self.bimamba_expand = getattr(args, "bimamba_expand", 1)
        self.input_size = args.embedding_dimension
        self.hidden_size = args.hidden_size
        self.projection = nn.Linear(
            args.embedding_dimension + args.hidden_size, args.target_dim
        )

        c_scaling = self.channel_mult ** (self.num_preprocess_blocks)  # 4
        spatial_scaling = 2 ** (self.num_preprocess_blocks)  # 4

        prior_ftr0_size = (
            int(c_scaling * self.num_channels_dec),
            args.prediction_length // spatial_scaling,
            (args.embedding_dimension + args.hidden_size + 1) // spatial_scaling,
        )
        self.prior_ftr0 = nn.Parameter(
            torch.rand(size=prior_ftr0_size), requires_grad=True
        )

        self.pre_process = self.init_pre_process(args.mult)
        self.enc_tower = self.init_encoder_tower(self.mult)

        self.enc0 = nn.Sequential(
            nn.ELU(),
            Conv2D(
                self.num_channels_enc * self.mult,
                self.num_channels_enc * self.mult,
                kernel_size=1,
                bias=True,
            ),
            nn.ELU(),
        )

        self.enc_sampler, self.dec_sampler = self.init_sampler(self.mult)

        self.dec_tower = self.init_decoder_tower(self.mult)

        self.post_process = self.init_post_process(self.mult)
        self.image_conditional = nn.Sequential(
            nn.ELU(),
            Conv2D(int(self.num_channels_dec * self.mult), 2, 3, padding=1, bias=True),
        )

    def _make_normal_cell(self, num_c, cell_type, arch):
        if self.use_bimamba:
            return CellMamba(
                int(num_c), cell_type,
                d_state=self.bimamba_d_state,
                expand=self.bimamba_expand,
            )
        return Cell(num_c, num_c, cell_type=cell_type, arch=arch, use_se=self.use_se)

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance["down_pre"]
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(self.channel_mult * num_ci)
                    cell = Cell(
                        num_ci,
                        num_co,
                        cell_type="down_pre",
                        arch=arch,
                        use_se=self.use_se,
                    )
                    mult = self.channel_mult * mult
                else:
                    num_c = self.num_channels_enc * mult
                    cell = self._make_normal_cell(
                        num_c, "normal_pre", self.arch_instance["normal_pre"]
                    )
                pre_process.append(cell)
        self.mult = mult
        return pre_process

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_enc * mult)
            cell = self._make_normal_cell(
                num_c, "normal_enc", self.arch_instance["normal_enc"]
            )
            enc_tower.append(cell)

            if not (g == self.groups_per_scale - 1):
                num_ce = int(self.num_channels_enc * mult)
                num_cd = int(self.num_channels_dec * mult)
                cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type="combiner_enc")
                enc_tower.append(cell)

        self.mult = mult
        return enc_tower

    def init_decoder_tower(self, mult):
        dec_tower = nn.ModuleList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_dec * mult)
            if not (g == 0):
                cell = self._make_normal_cell(
                    num_c, "normal_dec", self.arch_instance["normal_dec"]
                )
                dec_tower.append(cell)
            cell = DecCombinerCell(
                num_c, self.num_latent_per_group, num_c, cell_type="combiner_dec"
            )
            dec_tower.append(cell)
        self.mult = mult
        return dec_tower

    def init_sampler(self, mult):
        enc_sampler = nn.ModuleList()
        dec_sampler = nn.ModuleList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_enc * mult)
            cell = Conv2D(
                num_c,
                2 * self.num_latent_per_group,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            enc_sampler.append(cell)
            if g != 0:
                num_c = int(self.num_channels_dec * mult)
                cell = nn.Sequential(
                    nn.ELU(),
                    Conv2D(
                        num_c,
                        2 * self.num_latent_per_group,
                        kernel_size=1,
                        padding=0,
                        bias=True,
                    ),
                )
                dec_sampler.append(cell)
        mult = mult / self.channel_mult
        return enc_sampler, dec_sampler

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance["up_post"]
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / self.channel_mult)
                    cell = Cell(
                        num_ci,
                        num_co,
                        cell_type="up_post",
                        arch=arch,
                        use_se=self.use_se,
                    )
                    mult = mult / self.channel_mult
                else:
                    num_c = int(self.num_channels_dec * mult)
                    cell = self._make_normal_cell(
                        num_c, "normal_post", self.arch_instance["normal_post"]
                    )
                post_process.append(cell)
        self.mult = mult
        return post_process

    def forward(self, x):
        s = self.stem(x)
        for cell in self.pre_process:
            s = cell(s)
        combiner_cells_enc = []
        combiner_cells_s = []
        kl_all = []
        for cell in self.enc_tower:
            if cell.cell_type == "combiner_enc":
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        ftr = self.enc0(s)
        param0 = self.enc_sampler[0](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)
        z, _ = dist.sample()
        prior0 = Normal(torch.zeros_like(mu_q), torch.zeros_like(log_sig_q))
        kl_all.append(torch.sum(dist.kl(prior0), dim=[1, 2, 3]))

        s = self.prior_ftr0.unsqueeze(0).expand(z.size(0), -1, -1, -1)
        idx_dec = 0
        for cell in self.dec_tower:
            if cell.cell_type == "combiner_dec":
                if idx_dec > 0:
                    # Prior p(z_n) a partir de las features del decoder (dec_sampler)
                    prior_param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(prior_param, 2, dim=1)
                    prior = Normal(mu_p, log_sig_p)
                    # Posterior q(z_n) a partir de las features del encoder combinadas
                    ftr = combiner_cells_enc[idx_dec - 1](
                        combiner_cells_s[idx_dec - 1], s
                    )
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()
                    kl_all.append(torch.sum(dist.kl(prior), dim=[1, 2, 3]))
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.post_process:
            s = cell(s)
        logits = self.image_conditional(s)
        logits = self.projection(logits[..., -(self.input_size + self.hidden_size) :])

        self.kl_loss = torch.stack(kl_all, dim=0).sum(dim=0)
        return logits

    def decoder_output(self, logits):
        return NormalDecoder(logits)
