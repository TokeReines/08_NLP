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

        x = self.model(tokens[:, :self.max_len], attention_mask=token_mask[:, :self.max_len].float())[-1]
        x = self.scalar_mix(x[-self.n_layers:])
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
    
class TransformerEmbeddingQuesDoc(nn.Module):
    def __init__(
        self,
        name: str,
        n_layers: int,
        n_out: int = 0,
        stride: int = 256,
        pooling: str = 'mean',
        pad_index: int = 0,
        mix_dropout: float = .0,
        finetune: bool = False,
        dropout: int = 0.2
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
        self.doc_stride = stride
        self.doc_max_len = 384

        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, d_tokens: torch.Tensor) -> torch.Tensor:
        q_mask = q_tokens.ne(self.pad_index)
        q_lens = q_mask.sum((1, 2))
        d_mask = d_tokens.ne(self.pad_index)
        d_lens = d_mask.sum((1, 2))
        batch, ques_len, fix_len_q = q_mask.shape
        _, doc_len, fix_len_d = d_mask.shape
        if fix_len_d > fix_len_q:
            t = q_tokens.new_zeros(batch, ques_len, fix_len_d)
            t[:, :, :fix_len_q] = q_tokens
            t[:, :, fix_len_q:] = self.pad_index
            q_tokens = t
        if fix_len_d < fix_len_q:
            t = d_tokens.new_zeros(batch, doc_len, fix_len_q)
            t[:, :, :fix_len_d] = d_tokens
            t[:, :, fix_len_d:] = self.pad_index
            d_tokens = t
        tokens = []
        batch_id = []

        eos = d_tokens.new_zeros(max(fix_len_q, fix_len_d))
        eos[0] = torch.tensor(self.tokenizer.vocab[self.tokenizer.eos])
        eos = eos.unsqueeze(0)

        for i in range(batch):
            q_i = q_tokens[i]
            d_i = d_tokens[i]
            d_mask_i = d_i.ne(self.pad_index).any(-1)
            d_i = d_i[d_mask_i]
            new_batch = torch.cat([q_i, d_i[:self.doc_max_len], eos], 0)
            batch_id.append(i)
            tokens.append(new_batch)
            for j in range(self.doc_stride, (d_i.shape[0]-self.doc_max_len+self.doc_stride-1)//self.doc_stride*self.doc_stride+1, self.doc_stride):
                new_batch = torch.cat([q_i, d_i[j:j+self.doc_max_len], eos], 0)
                tokens.append(new_batch)
                batch_id.append(i)
        batch_id = torch.tensor(batch_id, device=q_tokens.get_device()).detach()
        tokens = pad(tokens, self.pad_index).detach()

        mask = tokens.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_tokens]
        tokens = pad(tokens[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        token_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        x = self.model(tokens[:, :self.max_len], attention_mask=token_mask[:, :self.max_len].float())[-1]
        x = x[-1]

        for i in range(self.stride, (tokens.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.model(tokens[:, i:i+self.max_len], attention_mask=token_mask[:, i:i+self.max_len].float())[-1]
            x = torch.cat((x, part[-1][:, self.max_len-self.stride:]), 1)
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

        x = self.projection(x)
        x = self.dropout(x)

        return x[:, :ques_len, :], x[:, ques_len:, :], batch_id, mask.any(-1)[:, :ques_len], mask.any(-1)[:, ques_len:]