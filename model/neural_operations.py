# -*-Encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict


BN_EPS = 1e-5
SYNC_BN = False

OPS = OrderedDict([
    ('res_elu', lambda Cin, Cout, stride: ELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnelu', lambda Cin, Cout, stride: BNELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish5', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 2, 2)),
    ('mconv_e6k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=1)),
    ('mconv_e3k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=1)),
    ('mconv_e3k5g8', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=8)),
    ('mconv_e6k11g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=11, g=0)),
    # BI-Mamba drop-in: reemplaza cada op conv individual dentro de Cell
    ('mamba_op', lambda Cin, Cout, stride: MambaOp(Cin, Cout)),
])


class SyncBatchNormSwish(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        super(SyncBatchNormSwish, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        self.ddp_gpu_size = None

    def forward(self, input):
        exponential_average_factor = self.momentum
        out = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        return out


def get_skip_connection(C, stride, channel_mult):
    if stride == 1:
        return Identity()
    elif stride == 2:
        return FactorizedReduce(C, int(channel_mult * C))
    elif stride == -1:
        return nn.Sequential(UpSample(), Conv2D(C, int(C / channel_mult), kernel_size=1))


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


def logit(t):
    return torch.log(t) - torch.log(1 - t)


def act(t):
    # The following implementation has lower memory.
    return SwishFN.apply(t)


class SwishFN(torch.autograd.Function):
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return act(x)


def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2, 3]))   # norm(w)
    weight = n * weight / (wn.view(-1, 1, 1, 1) + 1e-5)
    return weight


class Conv2D(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False,
                 weight_norm=True):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(Conv2D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.log_weight_norm = None
        if weight_norm:
            init = norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1)
            self.log_weight_norm = nn.Parameter(torch.log(init + 1e-2), requires_grad=True)

        self.data_init = data_init
        self.init_done = False
        self.weight_normalized = self.normalize_weight()

    def forward(self, x):
        # do data based initialization
        self.weight_normalized = self.normalize_weight()
        #print(self.weight_normalized.shape)
        bias = self.bias
        return F.conv2d(x, self.weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def normalize_weight(self):
        """ applies weight normalization """
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        else:
            weight = self.weight

        return weight


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SyncBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SyncBatchNorm, self).__init__()
        self.bn = nn.BatchNorm(*args, **kwargs)

    def forward(self, x):
        return self.bn(x)


# quick switch between multi-gpu, single-gpu batch norm
def get_batchnorm(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


class ELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(ELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation,
                             data_init=True)

    def forward(self, x):
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn = get_batchnorm(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNSwishConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act = SyncBatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = self.bn_act(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = Conv2D(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = act(x)
        conv1 = self.conv_1(out[:,:,:, :])
        conv2 = self.conv_2(out[:, :, 1:, :])
        conv3 = self.conv_3(out[:, :, :, :])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class EncCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        # Cin = Cin1 + Cin2
        self.conv = Conv2D(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class ConvBNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv2D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False),
            SyncBatchNormSwish(Cout, eps=BN_EPS, momentum=0.05)  # drop in replacement for BN + Swish
        )

    def forward(self, x):
        return self.conv(x)


class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class InvertedResidual(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g

        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [get_batchnorm(Cin, eps=BN_EPS, momentum=0.05),
                  ConvBNSwish(Cin, hidden_dim, k=1),
                  ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                  Conv2D(hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False),
                  get_batchnorm(Cout, momentum=0.05)]

        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x):
        return self.conv(x)


class MambaOp(nn.Module):
    """
    Drop-in replacement for a single conv op inside Cell (e.g. BNSwishConv or InvertedResidual).

    Recibe (B, C, H, W) igual que cualquier op en OPS.
    Opera sobre el eje H (dimension temporal) tratando C como la dimension de embedding.
    Aplica Mamba bidireccional: forward (H[0]→H[-1]) + backward (H[-1]→H[0]).
    Devuelve (B, C, H, W) con el mismo shape.

    Flujo interno:
        (B, C, H, W)
        → permute → (B, W, H, C)          # W posiciones tratadas en batch
        → reshape  → (B*W, H, C)           # secuencia de largo H, embedding C
        → Mamba_fwd(B*W, H, C)  +  Mamba_bwd(B*W, H, C) flipped
        → suma + residual
        → LayerNorm(C)
        → FFN: Linear(C→C*2)→SiLU→Linear(C*2→C)
        → LayerNorm(C)
        → reshape → (B, W, H, C)
        → permute → (B, C, H, W)
    """

    def __init__(self, Cin, Cout):
        super().__init__()
        # MambaOp solo soporta stride=1 (las ops normal_enc/normal_dec son stride=1)
        # Si Cin != Cout necesitamos adaptar canales primero
        self.adapt = nn.Identity() if Cin == Cout else nn.Conv2d(Cin, Cout, kernel_size=1, bias=False)
        C = Cout

        # Mamba SSM params — valores conservadores para caber en GPU pequeña
        d_state = 16          # dimension del estado SSM
        d_conv  = 3           # kernel causal
        d_inner = max(C // 2, 8)  # sin expansion; d_inner < C para reducir memoria

        # Forward SSM
        self.fwd_in    = nn.Linear(C, d_inner * 2, bias=False)
        self.fwd_conv  = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner, bias=True)
        self.fwd_xproj = nn.Linear(d_inner, 1 + d_state * 2, bias=False)
        self.fwd_dt    = nn.Linear(1, d_inner, bias=True)
        A_fwd = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.fwd_A_log = nn.Parameter(torch.log(A_fwd))
        self.fwd_D     = nn.Parameter(torch.ones(d_inner))
        self.fwd_out   = nn.Linear(d_inner, C, bias=False)

        # Backward SSM (pesos independientes)
        self.bwd_in    = nn.Linear(C, d_inner * 2, bias=False)
        self.bwd_conv  = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner, bias=True)
        self.bwd_xproj = nn.Linear(d_inner, 1 + d_state * 2, bias=False)
        self.bwd_dt    = nn.Linear(1, d_inner, bias=True)
        A_bwd = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.bwd_A_log = nn.Parameter(torch.log(A_bwd))
        self.bwd_D     = nn.Parameter(torch.ones(d_inner))
        self.bwd_out   = nn.Linear(d_inner, C, bias=False)

        # Post-processing
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(
            nn.Linear(C, C * 2),
            nn.SiLU(),
            nn.Linear(C * 2, C),
        )

    def _ssm_branch(self, x, in_proj, conv1d, xproj, dt_proj, A_log, D, out_proj):
        """Aplica un brazo SSM sobre x: (BW, L, C) → (BW, L, C)."""
        BW, L, C = x.shape
        d_inner = in_proj.out_features // 2
        d_state = (xproj.out_features - 1) // 2

        # 1) proyeccion entrada
        xz  = in_proj(x)                                  # (BW, L, d_inner*2)
        xi, gate = xz.split(d_inner, dim=-1)

        # 2) conv causal sobre L
        xi = F.silu(conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2))  # (BW, L, d_inner)

        # 3) SSM proyection: delta (1), B (d_state), C_ssm (d_state)
        dBC   = xproj(xi)                                 # (BW, L, 1+2*d_state)
        delta = F.softplus(dt_proj(dBC[..., :1]))         # (BW, L, d_inner)
        B_ssm = dBC[..., 1:1+d_state]                    # (BW, L, d_state)
        C_ssm = dBC[..., 1+d_state:]                     # (BW, L, d_state)

        # 4) Selective scan discreto
        A  = -torch.exp(A_log.float())                    # (d_inner, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))       # (BW,L,d_inner,d_state)
        dBu = torch.einsum('bld,bln,bld->bldn', delta, B_ssm, xi)   # (BW,L,d_inner,d_state)

        h = torch.zeros(BW, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dBu[:, i]
            y = torch.einsum('bdn,bn->bd', h, C_ssm[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)                        # (BW, L, d_inner)
        y = y + xi * D

        # 5) gate + salida
        return out_proj(y * F.silu(gate))                 # (BW, L, C)

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)"""
        x = self.adapt(x)
        B, C, H, W = x.shape

        # (B, C, H, W) → (B*W, H, C): W como batch extra, H como secuencia, C como embedding
        seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)

        # brazo forward
        y_fwd = self._ssm_branch(seq,
            self.fwd_in, self.fwd_conv, self.fwd_xproj,
            self.fwd_dt, self.fwd_A_log, self.fwd_D, self.fwd_out)

        # brazo backward: invertir H, SSM, volver a invertir
        y_bwd = self._ssm_branch(seq.flip(1),
            self.bwd_in, self.bwd_conv, self.bwd_xproj,
            self.bwd_dt, self.bwd_A_log, self.bwd_D, self.bwd_out).flip(1)

        # combinar + residual + LayerNorm
        y = self.norm1(y_fwd + y_bwd + seq)

        # FFN + residual + LayerNorm
        y = self.norm2(y + self.ffn(y))

        # (B*W, H, C) → (B, W, H, C) → (B, C, H, W)
        return y.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
