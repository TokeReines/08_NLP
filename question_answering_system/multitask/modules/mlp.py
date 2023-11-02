from __future__ import annotations

import torch
import torch.nn as nn

from modules.dropout import SharedDropout

class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = .0, activation: bool = True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x