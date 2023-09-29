
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
    