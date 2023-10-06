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

    def update(self, y, gold):
        mask = gold.ge(0)
        y = torch.flatten(y[mask])
        gold = torch.flatten(gold[mask])
        pos = gold > 0
        self.true_pos += gold[pos].eq(y[pos]).sum()
        self.false_neg += gold[pos].ne(y[pos]).sum()
        self.false_pos += y[y > 0].ne(gold[y > 0]).sum()
    
    @property
    def score(self):
        return 2 * self.true_pos / (2 * self.true_pos + self.false_pos + self.false_neg + self.eps)
    
class F1Metric2():
    def __init__(self, eps=1e-12):
        self.eps = eps
        # self.true_pos = 0.0
        # self.false_pos = 0.0
        # self.false_neg = 0.0
        self.pred = np.array([])
        self.gold = np.array([])

    def update(self, y, gold, y_answerable):
        y_answerable = y_answerable.eq(1)
        y[y_answerable] = 0.0
        mask = gold.ge(0)
        y = torch.flatten(y[mask]).cpu().numpy()
        gold = torch.flatten(gold[mask]).cpu().numpy()
        # pos = gold > 0
        # self.true_pos += gold[pos].eq(y[pos]).sum()
        # self.false_neg += gold[pos].ne(y[pos]).sum()
        # self.false_pos += y[y > 0].ne(gold[y > 0]).sum()
        self.pred = np.hstack([self.pred, y])
        self.gold = np.hstack([self.gold, gold])

    @property
    def score(self):
        # return 2 * self.true_pos / (2 * self.true_pos + self.false_pos + self.false_neg + self.eps)
        return f1_score(self.gold, self.pred)