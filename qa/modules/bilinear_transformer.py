import torch
import torch.nn as nn

class HeadAttention(nn.Module):

    def __init__(
        self,
        n_heads: int = 8,
        n_model: int = 1024,
        n_embed: int = 128,
        dropout: float = 0.1,
        bias: bool = True,
        attn: bool = False,
    ):
        super(HeadAttention, self).__init__()

        pass