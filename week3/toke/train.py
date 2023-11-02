
import torch_directml
from collections import Counter
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from week3.toke.ffn.dataset_bow import DatasetBOW
from week3.toke.ffn.ffn import FeedForwardNetwork
from week3.toke.ffn.tokenizer_bow import TokenizerBOW
from week3.toke.ffn.transform import TransformBOW

from week3.toke.trainer import Trainer
from week3.toke.bow_dataset import BOWDataset
from week3.toke.tokenizer import Tokenizer
from week3.toke.transform import Transform
from week3.toke.ffn import FFN


train_set = pd.read_feather('week3/toke/data/arabic_train.feather')
dev_set = pd.read_feather('week3/toke/data/arabic_dev.feather')
test_set = pd.read_feather('week3/toke/data/arabic_test.feather')

# batch_size = 16

# tokenizer = Tokenizer(train_set)
# transform = Transform(tokenizer)

# train_dataset = BOWDataset(tokenizer, transform, train_set)
# dev_set = BOWDataset(tokenizer, transform, dev_set)
# test_set = BOWDataset(tokenizer, transform, test_set)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
# dev_dataloader = DataLoader(dev_set, batch_size=batch_size)
# test_dataloader = DataLoader(test_set, batch_size=batch_size)

# model = FFN(len(tokenizer.vocab)*2+1, 128, 1)
# optimizer = optim.SGD(model.parameters(), lr=0.000001)

# #dml = torch_directml.device()
# trainer = Trainer('week3/toke/models/ffn.pt', model, 'cpu', tokenizer)
# trainer.train(optimizer, 50, train_dataloader, dev_dataloader)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Define the Tokenizer class
class Tokenizer():
    def __init__(self, separator="<SEP>"):
        self.separator = separator
        self.vocab = {separator: 0, '<UNK>': 1, '<PAD>': 2}
    
    def __call__(self, text):
        tokens = text.split()
        for token in tokens:
            if token in self.vocab:
                continue
            
            self.vocab[token] = len(self.vocab)
            
        return tokens
    
# Define the Transform class
class Transform():
    def __init__(self, tokenizer, max_length=300):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def pad(self, tokens, max_length):
        tokens = list(tokens)
        if len(tokens) < max_length:
            tokens += ['<PAD>'] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            
        return tokens

    def __call__(self, data: list[list[str]]):      
        padded = [self.pad(d, self.max_length) for d in data]  
        tokens = [self.tokenizer(" ".join(p)) for p in padded]
        
        return tokens

# Define the custom Dataset class
class ArabicDataset(Dataset):
    def __init__(self, data, transform, tokenizer):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.chunk()
        
    def chunk(self):
        """
        Transform data and chunk it into a list of tuples
        """
        ques, answerable, doc = list(self.data['question_text']), list(self.data['answerable']), list(self.data['document_plaintext'])

        ques = self.transform(ques)
        doc = self.transform(doc)

        self.data = list(zip(ques, answerable, doc))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answerable, document = self.data[idx]
        question = [self.tokenizer.vocab[t] for t in question]
        document = [self.tokenizer.vocab[t] for t in document]
        return torch.tensor(question), torch.tensor(answerable), torch.tensor(document)
        






max_length = 3000
# Example usage
tokenizer = TokenizerBOW()
transform = TransformBOW(tokenizer, max_length=max_length)

train_set = pd.read_feather('week3/toke/data/arabic_train.feather')
dev_set = pd.read_feather('week3/toke/data/arabic_dev.feather')
test_set = pd.read_feather('week3/toke/data/arabic_test.feather')

train_dataset = DatasetBOW(train_set, transform, tokenizer, max_length)
dev_dataset = DatasetBOW(dev_set, transform, tokenizer, max_length)
test_dataset = DatasetBOW(test_set, transform, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

embedding_dim = 100
hidden_dim = 50
model = FeedForwardNetwork(max_length*2+1, hidden_dim, 1)
optimizer = optim.SGD(model.parameters(), lr=0.000001)

dml = torch_directml.device()
trainer = Trainer('week3/toke/models/ffn.pt', model, dml, tokenizer)
trainer.train(optimizer, 50, train_dataloader, dev_dataloader)