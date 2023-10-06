
class Metric(object):
    def __init__(self, eps = 1e-12):
        super.__init__()
        self.eps = eps
    
    def __lt__(self, other):
        return self.score < other.score 
    
    def __le__(self, other):
        return self.score <= other.score
    
    def __gt__(self, other):
        return self.score > other.score
    
    def __ge__(self, other):
        return self.score >= other.score
    
    def __add__(self, other):
        return other

    @property
    def score(self):
        raise AttributeError

class AccuracyMetric(Metric):
    def __init__(self, preds, golds, eps):
        super.__init__(eps)
        self.total =  0.0
        self.correct = 0.0
    
    def __call__(self, preds, golds):
        acc_mask = preds.eq(golds)
        self.count += 1
        self.total += preds.shape(0)
        self.correct += acc_mask.sum().item()
        return self

    @property 
    def score(self):
        return self.correct / (self.total + self.eps)
    
