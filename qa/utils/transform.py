

class BertTransform():
    def __init__(self, tokenizer, fix_len=0):
        self.fix_len = fix_len
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

    def __call__(self, data):
        res = []
        for seq in data:
            seq = [self.tokenizer(token) for token in seq]
            seq = [[self.vocab[i] if i in self.vocab else self.vocab[self.tokenizer.unk] for i in token] if token else [self.vocab[self.tokenizer.unk]] for token in seq]
            seq = [[self.vocab[self.tokenizer.bos]]] + seq
            seq = seq + [[self.vocab[self.tokenizer.eos]]]
            if self.fix_len > 0:
                seq = [ids[:self.fix_len] for ids in seq]
            res.append(seq)
        return res
    
