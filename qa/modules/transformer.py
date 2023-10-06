# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerWordEmbedding(nn.Module):

    def __init__(
        self,
        n_vocab: int = None,
        n_embed: int = None,
        embed_scale: Optional[int] = None,
        max_len: Optional[int] = 512,
        pos: Optional[str] = None,
        pad_index: Optional[int] = None,
    ) -> TransformerWordEmbedding:
        super(TransformerWordEmbedding, self).__init__()

        self.embed = nn.Embedding(num_embeddings=n_vocab,
                                  embedding_dim=n_embed)

        self.pos_embed = PositionalEmbedding(max_len=max_len)

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.embed_scale = embed_scale or n_embed ** 0.5
        self.max_len = max_len
        self.pos = pos
        self.pad_index = pad_index

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_vocab}, {self.n_embed}"
        if self.embed_scale is not None:
            s += f", embed_scale={self.embed_scale:.2f}"
        if self.max_len is not None:
            s += f", max_len={self.max_len}"
        if self.pos is not None:
            s += f", pos={self.pos}"
        if self.pad_index is not None:
            s += f", pad_index={self.pad_index}"
        s += ')'
        return s

    def reset_parameters(self):
        nn.init.normal_(self.embed.weight, 0, self.n_embed ** -0.5)
        if self.pad_index is not None:
            nn.init.zeros_(self.embed.weight[self.pad_index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        if self.embed_scale:
            x = x * self.embed_scale
        if self.pos is not None:
            x = x + self.pos_embed(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        layer: nn.Module,
        n_layers: int = 6,
        n_model: int = 1024,
        pre_norm: bool = False,
    ) -> TransformerEncoder:
        super(TransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_model = n_model
        self.pre_norm = pre_norm

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, mask)
        if self.pre_norm:
            x = self.norm(x)
        return x.transpose(0, 1)


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        layer: nn.Module,
        n_layers: int = 6,
        n_model: int = 1024,
        pre_norm: bool = False,
    ) -> TransformerDecoder:
        super(TransformerDecoder, self).__init__()

        self.n_layers = n_layers
        self.n_model = n_model
        self.pre_norm = pre_norm

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        x_tgt, x_src = x_tgt.transpose(0, 1), x_src.transpose(0, 1)
        for layer in self.layers:
            x_tgt = layer(x_tgt=x_tgt,
                          x_src=x_src,
                          tgt_mask=tgt_mask,
                          src_mask=src_mask,
                          attn_mask=attn_mask)
        if self.pre_norm:
            x_tgt = self.norm(x_tgt)
        return x_tgt.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        activation: str = 'relu',
        bias: bool = True,
        pre_norm: bool = False,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        dropout: float = 0.1
    ) -> TransformerEncoderLayer:
        super(TransformerEncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(n_heads=n_heads,
                                       n_model=n_model,
                                       n_embed=n_model//n_heads,
                                       dropout=attn_dropout,
                                       bias=bias)
        self.attn_norm = nn.LayerNorm(n_model)
        self.ffn = PositionwiseFeedForward(n_model=n_model,
                                           n_inner=n_inner,
                                           activation=activation,
                                           dropout=ffn_dropout)
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.pre_norm:
            n = self.attn_norm(x)
            x = x + self.dropout(self.attn(n, n, n, mask))
            n = self.ffn_norm(x)
            x = x + self.dropout(self.ffn(n))
        else:
            x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask)))
            x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        activation: str = 'relu',
        bias: bool = True,
        pre_norm: bool = False,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        dropout: float = 0.1
    ) -> TransformerDecoderLayer:
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_heads=n_heads,
                                            n_model=n_model,
                                            n_embed=n_model//n_heads,
                                            dropout=attn_dropout,
                                            bias=bias)
        self.self_attn_norm = nn.LayerNorm(n_model)
        self.mha_attn = MultiHeadAttention(n_heads=n_heads,
                                           n_model=n_model,
                                           n_embed=n_model//n_heads,
                                           dropout=attn_dropout,
                                           bias=bias)
        self.mha_attn_norm = nn.LayerNorm(n_model)
        self.ffn = PositionwiseFeedForward(n_model=n_model,
                                           n_inner=n_inner,
                                           activation=activation,
                                           dropout=ffn_dropout)
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            n_tgt = self.self_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.self_attn(n_tgt, n_tgt, n_tgt, tgt_mask, attn_mask))
            n_tgt = self.mha_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.mha_attn(n_tgt, x_src, x_src, src_mask))
            n_tgt = self.ffn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.ffn(x_tgt))
        else:
            x_tgt = self.self_attn_norm(x_tgt + self.dropout(self.self_attn(x_tgt, x_tgt, x_tgt, tgt_mask, attn_mask)))
            x_tgt = self.mha_attn_norm(x_tgt + self.dropout(self.mha_attn(x_tgt, x_src, x_src, src_mask)))
            x_tgt = self.ffn_norm(x_tgt + self.dropout(self.ffn(x_tgt)))
        return x_tgt

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads: int = 8,
        n_model: int = 1024,
        n_embed: int = 128,
        dropout: float = 0.1,
        bias: bool = True,
        attn: bool = False,
    ) -> MultiHeadAttention:
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed
        self.scale = n_embed**0.5

        self.wq = nn.Linear(n_model, n_heads * n_embed, bias=bias)
        self.wk = nn.Linear(n_model, n_heads * n_embed, bias=bias)
        self.wv = nn.Linear(n_model, n_heads * n_embed, bias=bias)
        self.wo = nn.Linear(n_heads * n_embed, n_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.bias = bias
        self.attn = attn

        self.reset_parameters()

    def reset_parameters(self):
        # borrowed from https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.wq.weight, 2 ** -0.5)
        nn.init.xavier_uniform_(self.wk.weight, 2 ** -0.5)
        nn.init.xavier_uniform_(self.wv.weight, 2 ** -0.5)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        batch_size, _ = mask.shape
        # [seq_len, batch_size * n_heads, n_embed]
        q = self.wq(q).view(-1, batch_size * self.n_heads, self.n_embed)
        k = self.wk(k).view(-1, batch_size * self.n_heads, self.n_embed)
        v = self.wv(v).view(-1, batch_size * self.n_heads, self.n_embed)

        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1).view(-1, 1, *mask.shape[1:])
        # [batch_size * n_heads, seq_len, src_len]
        if attn_mask is not None:
            mask = mask & attn_mask
        # [batch_size * n_heads, seq_len, src_len]
        attn = torch.bmm(q.transpose(0, 1) / self.scale, k.movedim((0, 1), (2, 0)))
        attn = torch.softmax(attn + torch.where(mask, 0., float('-inf')), -1)
        attn = self.dropout(attn)
        # [seq_len, batch_size * n_heads, n_embed]
        x = torch.bmm(attn, v.transpose(0, 1)).transpose(0, 1)
        # [seq_len, batch_size, n_model]
        x = self.wo(x.reshape(-1, batch_size, self.n_heads * self.n_embed))

        return (x, attn.view(batch_size, self.n_heads, *attn.shape[1:])) if self.attn else x

class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        n_model: int = 1024,
        n_inner: int = 2048,
        activation: str = 'relu',
        dropout: float = 0.1
    ) -> PositionwiseFeedForward:
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(n_model, n_inner)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(n_inner, n_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)

        return x

class PositionalEmbedding(nn.Embedding):

    def __init__(
        self,
        n_model: int = 1024,
        max_len: int = 1024
    ) -> PositionalEmbedding:
        super().__init__(max_len, n_model)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        w = self.weight
        max_len, n_model = w.shape
        w = w.new_tensor(range(max_len)).unsqueeze(-1)
        w = w / 10000 ** (w.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        w[:, 0::2], w[:, 1::2] = w[:, 0::2].sin(), w[:, 1::2].cos()
        self.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.weight, x.new_tensor(range(x.shape[1]), dtype=torch.long))