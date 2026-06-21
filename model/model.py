# -*-Encoding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from .resnet import Res12_Quadratic
from .diffusion_process import GaussianDiffusion, get_beta_schedule, extract
from .encoder import Encoder
from .embedding import DataEmbedding


class diffusion_generate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=args.embedding_dimension,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
            batch_first=True,
        )
        self.generative = Encoder(args)
        self.diffusion = GaussianDiffusion(
            self.generative,
            input_size=args.target_dim,
            diff_steps=args.diff_steps,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            scale=args.scale,
        )

    def forward(self, past_time_feat, future_time_feat, t):
        time_feat, _ = self.rnn(past_time_feat)
        x_in = torch.cat([time_feat, past_time_feat], dim=-1)
        output, y_noisy = self.diffusion.log_prob(x_in, future_time_feat, t)
        return output, y_noisy


class denoise_net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # ResNet that used to calculate the scores.
        self.score_net = Res12_Quadratic(1, 64, 32, normalize=False, AF=nn.ELU())

        # Peso sigma_n del DSM (Ec. 8): varianza de ruido del target en el paso n,
        # CONSISTENTE con el schedule usado para difundir el target en GaussianDiffusion
        # (alphas_target = 1 - beta*scale). El codigo original usaba un schedule
        # distinto (alphas = 1 - beta*0.5) que, ademas de no tener base en el paper,
        # colapsaba sigma_n ~ 1 para casi todo n cuando beta_end=1. Los buffers
        # alphas_cumprod / sqrt_* del original no se usaban (codigo muerto) y se
        # eliminan.
        betas = get_beta_schedule(
            args.beta_schedule, args.beta_start, args.beta_end, args.diff_steps
        )
        alphas_target_cumprod = np.cumprod(1.0 - betas * args.scale, axis=0)
        self.sigmas = torch.tensor(1.0 - alphas_target_cumprod, dtype=torch.float32)

        # The generative bvae model.
        self.diffusion_gen = diffusion_generate(args)

        # Data embedding module.
        self.embedding = DataEmbedding(
            args.input_dim,
            args.embedding_dimension,
            args.dropout_rate,
            temporal_embedding=args.temporal_embedding,
        )

    def forward(self, past_time_feat, mark, future_time_feat, t):
        x_embed = self.embedding(past_time_feat, mark)
        output, y_noisy = self.diffusion_gen(x_embed, future_time_feat, t)

        sigmas_t = extract(self.sigmas.to(y_noisy.device), t, y_noisy.shape)
        y = future_time_feat.unsqueeze(1).float()
        y_noisy1 = output.sample().float().requires_grad_()
        E = self.score_net(y_noisy1).sum()
        grad_x = torch.autograd.grad(E, y_noisy1, create_graph=True)[0]
        dsm_loss = torch.mean(
            torch.sum(((y - y_noisy1.detach()) + grad_x) ** 2 * sigmas_t, [1, 2, 3])
        ).float()
        return output, y_noisy, dsm_loss


class pred_net(denoise_net):
    def forward(self, x, mark):
        x_embed = self.embedding(x, mark)
        x_t, _ = self.diffusion_gen.rnn(x_embed)
        x_in = torch.cat([x_t, x_embed], dim=-1).unsqueeze(1)
        logits = self.diffusion_gen.generative(x_in)
        output = self.diffusion_gen.generative.decoder_output(logits)

        # Score correction needs autograd even during inference (inside torch.no_grad).
        # torch.enable_grad() overrides the outer no_grad context for just this block.
        # create_graph=False: we don't need second-order gradients during eval.
        with torch.enable_grad():
            y = output.mu.float().detach().requires_grad_(True)
            E = self.score_net(y).sum()
            grad_x = torch.autograd.grad(E, y)[0]

        out = (y - grad_x).detach()
        # sigma del decoder: incertidumbre aleatórica del VAE (usada para CRPS)
        sigma = output.sigma.detach()
        return y.detach(), out, sigma
