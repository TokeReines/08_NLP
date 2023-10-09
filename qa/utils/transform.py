
from utils.vocab import Vocab, build_vocabs
class BertTransform():
    def __init__(self, tokenizer, fix_len=0, lower_case=True, fn=None):
        self.fix_len = fix_len
        self.tokenizer = tokenizer
        self.lower_case = lower_case
        self.vocab = tokenizer.vocab
        self.fn = fn
    
    def preprocess(self, token):
        if self.lower_case:
            token = str.lower(token)
        if self.fn is not None:
            token = self.fn(token)
        return self.tokenizer(token)

    def __call__(self, data):
        res = []
        for seq in data:
            seq = [self.preprocess(token) for token in seq]
            seq = [[self.vocab[i] if i in self.vocab else self.vocab[self.tokenizer.unk] for i in token] if token else [self.vocab[self.tokenizer.unk]] for token in seq]
            seq = [[self.vocab[self.tokenizer.bos]]] + seq
            seq = seq + [[self.vocab[self.tokenizer.eos]]]
            if self.fix_len > 0:
                seq = [ids[:self.fix_len] for ids in seq]
            res.append(seq)
        return res
    
class WordTransform():
    def __init__(self, data, lower_case=True, fn=None, min_freq=2):
        self.lower_case = lower_case
        self.vocab = build_vocabs(data, min_freq)
        self.fn = fn
    
    def preprocess(self, token):
        if self.lower_case:
            token = str.lower(token)
        if self.fn is not None:
            token = self.fn(token)
        return self.vocab[token]

    def __call__(self, data):
        res = []
        for seq in data:
            seq = [self.preprocess(token) for token in seq]
            seq = [self.vocab['<bos>']] + seq
            seq = seq + [self.vocab['<eos>']]
            res.append(seq)
        return res