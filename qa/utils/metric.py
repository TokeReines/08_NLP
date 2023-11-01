import torch
from sklearn.metrics import f1_score
import numpy as np
class AccMetric():
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.count = 0.0
        self.n_correct = 0.0

    def update(self, y, gold, mask):
        # lens = mask.sum()
        score_mask = y.eq(gold)
        self.count += y.shape[0]
        self.n_correct += score_mask.sum()
    
    @property
    def score(self):
        return self.n_correct / (self.count + self.eps)
    
class F1Metric():
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.true_pos = 0.0
        self.false_pos = 0.0
        self.false_neg = 0.0
        self.total = 0.0
        self.sum_f1 = 0.0
        self.em = 0.0

    def update(self, y, gold, y_answerable):
        # y_answerable = y_answerable.eq(0)
        # y[y_answerable] = 0.0
        batch, _ = y.shape
        for i in range(batch):
            mask = gold[i, :].ge(0)
            y_i = torch.flatten(y[i, :][mask])
            gold_i = torch.flatten(gold[i, :][mask])
            pos = gold_i > 0
            true_pos = gold_i[pos].eq(y_i[pos]).sum()
            false_neg = gold_i[pos].ne(y_i[pos]).sum()
            false_pos = y_i[y_i > 0].ne(gold_i[y_i > 0]).sum()
            f1 = (2 * true_pos / (2 * true_pos + false_pos + false_neg + self.eps)).item()
            # f1 = f1_score(gold_i.cpu().numpy(), y_i.cpu().numpy())
            if f1 >= 1:
                self.em += 1
            self.sum_f1 += f1
        self.total += batch
        # mask = gold.ge(0)
        # y = torch.flatten(y[mask])
        # gold = torch.flatten(gold[mask])
        # pos = gold > 0
        # self.true_pos += gold[pos].eq(y[pos]).sum()
        # self.false_neg += gold[pos].ne(y[pos]).sum()
        # self.false_pos += y[y > 0].ne(gold[y > 0]).sum()
        
    
    @property
    def score(self):
        return self.sum_f1 / self.total, self.em / self.total

class F1Metric2():
    pass