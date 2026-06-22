# -*-Encoding: utf-8 -*-
import torch
import torch.nn as nn
from .encoder import Encoder
from .embedding import DataEmbedding


class vae_generate(nn.Module):
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

    def forward(self, past_time_feat, future_time_feat):
        time_feat, _ = self.rnn(past_time_feat)
        x_in = torch.cat([time_feat, past_time_feat], dim=-1)
        B, T, _ = x_in.shape
        x_in = x_in.reshape(B, 1, T, -1)
        logits = self.generative(x_in)
        output = self.generative.decoder_output(logits)
        B1, T1 = future_time_feat.shape[0], future_time_feat.shape[1]
        y = future_time_feat.reshape(B1, 1, T1, -1)
        return output, y


class vae_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vae_gen = vae_generate(args)
        self.embedding = DataEmbedding(
            args.input_dim,
            args.embedding_dimension,
            args.dropout_rate,
            temporal_embedding=args.temporal_embedding,
        )

    def forward(self, past_time_feat, mark, future_time_feat):
        x_embed = self.embedding(past_time_feat, mark)
        output, y = self.vae_gen(x_embed, future_time_feat)
        return output, y


class pred_net(vae_net):
    def forward(self, x, mark):
        x_embed = self.embedding(x, mark)
        x_t, _ = self.vae_gen.rnn(x_embed)
        x_in = torch.cat([x_t, x_embed], dim=-1).unsqueeze(1)
        logits = self.vae_gen.generative(x_in)
        output = self.vae_gen.generative.decoder_output(logits)
        mu = output.mu.float().detach()
        sigma = output.sigma.detach()
        return mu, mu, sigma
