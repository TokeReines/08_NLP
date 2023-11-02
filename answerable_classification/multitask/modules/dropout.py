from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

class SharedDropout(nn.Module):
    def __init__(self, p: float = 0.5, batch_first: bool = True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor):
        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)
    
    @staticmethod
    def get_mask(x: torch.Tensor, p: float):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)
    
class IndependentDropout(nn.Module):
    def _init__(self, p: float = 0.5):
        super().__init__()

        self.p = p
    
    def forward(self, *items: List[torch.Tensor]):
        if not self.training:
            return items
        masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
        total = sum(masks)
        scale = len(items) / total.max(torch.ones_like(total))
        masks = [mask * scale for mask in masks]
        return [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)] 