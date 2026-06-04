# -*-Encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Encoder
from .embedding import DataEmbedding


class core_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.target_dim = args.target_dim
        self.input_size = args.embedding_dimension
        self.prediction_length = args.prediction_length
        self.seq_length = args.sequence_length
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
            batch_first=True,
        )
        self.generative = Encoder(args)
        self.projection = nn.Linear(args.embedding_dimension+args.hidden_size, args.embedding_dimension)

    def forward(self, past_time_feat):
        time_feat, _ = self.rnn(past_time_feat)
        input = torch.cat([time_feat, past_time_feat], dim=-1)
        input = input.unsqueeze(1)
        logits = self.generative(input)
        output = self.generative.decoder_output(logits)
        return output


class train_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.core = core_model(args)
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension, args.dropout_rate)

    def forward(self, past_time_feat, mark):
        input = self.embedding(past_time_feat, mark)
        output = self.core(input)
        return output


class pred_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.core = core_model(args)
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension, args.dropout_rate)

    def forward(self, x, mark):
        input = self.embedding(x, mark)
        output = self.core(input)
        return output, output.mu
