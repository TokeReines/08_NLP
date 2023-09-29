from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from utils.fn import pad
from utils.tokenizer import TransformerTokenizer

class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        name: str,
        n_layers: int,
        n_out: int = 0,
        stride: int = 256,
        pooling: str = 'mean',
        pad_index: int = 0,
        mix_dropout: float = .0,
        finetune: bool = False
    ):
        super().__init__()

        from transformers import AutoModel
        try:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=True)
        except Exception:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=False)
        self.model = self.model.requires_grad_(finetune)
        self.tokenizer = TransformerTokenizer(name)

        self.name = name
        self.n_layers = n_layers or self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pooling = pooling
        self.pad_index = pad_index
        self.mix_dropout = mix_dropout
        self.finetune = finetune
        self.max_len = int(max(0, self.model.config.max_position_embeddings) or 1e12) - 2
        self.stride = min(stride, self.max_len)

        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_tokens]
        tokens = pad(tokens[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        token_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        x = self.model(tokens[:, :self.max_len], attention_mask=token_mask[:, :self.max_len].float())[-1]
        # [batch_size, max_len, hidden_size]
        x = self.scalar_mix(x[-self.n_layers:])
        # [batch_size, n_tokens, hidden_size]
        for i in range(self.stride, (tokens.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.model(tokens[:, i:i+self.max_len], attention_mask=token_mask[:, i:i+self.max_len].float())[-1]
            x = torch.cat((x, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        lens = lens.masked_fill_(lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        x = x.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), x[token_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            x = x[:, :, 0]
        elif self.pooling == 'last':
            x = x.gather(2, (lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        elif self.pooling == 'mean':
            x = x.sum(2) / lens.unsqueeze(-1)
        elif self.pooling:
            raise RuntimeError(f'Unsupported pooling method "{self.pooling}"!')
        return self.projection(x)
    
class ScalarMix(nn.Module):
    def __init__(self, n_layers: int, dropout: float = .0) -> ScalarMix:
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return self.gamma * sum(w * h for w, h in zip(self.dropout(self.weights.softmax(-1)), tensors))