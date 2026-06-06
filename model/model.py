# -*-Encoding: utf-8 -*-
import torch
import torch.nn as nn
from .encoder import Encoder
from .embedding import DataEmbedding


class pred_net(nn.Module):
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
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension, args.dropout_rate)

    def forward(self, x, mark):
        input = self.embedding(x, mark)
        x_t, _ = self.rnn(input)
        input = torch.cat([x_t, input], dim=-1)
        input = input.unsqueeze(1)
        logits = self.generative(input)
        output = self.generative.decoder_output(logits)
        return output
