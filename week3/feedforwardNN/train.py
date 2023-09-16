
import torch
import torch_directml
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from week3.feedforwardNN.trainer import Trainer
from week3.feedforwardNN.bow_dataset import BOWDataset
from week3.feedforwardNN.tokenizer import Tokenizer
from week3.feedforwardNN.transform import Transform
from week3.feedforwardNN.ffn import FFN

n = 500

train_set = pd.read_feather('week3/feedforwardNN/data/arabic_train.feather')
dev_set = pd.read_feather('week3/feedforwardNN/data/arabic_dev.feather')
test_set = pd.read_feather('week3/feedforwardNN/data/arabic_test.feather')

batch_size = 16

tokenizer = Tokenizer()
transform = Transform(tokenizer)

train_dataset = BOWDataset(tokenizer, transform, train_set)
dev_set = BOWDataset(tokenizer, transform, dev_set)
test_set = BOWDataset(tokenizer, transform, test_set)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
dev_dataloader = DataLoader(dev_set, batch_size=batch_size)
test_dataloader = DataLoader(test_set, batch_size=batch_size)

model = FFN(len(tokenizer.vocab)*2, 128, 1)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

dml = torch_directml.device()
trainer = Trainer('week3/feedforwardNN/models/ffn.pt', model, dml)
trainer.train(optimizer, 50, train_dataloader, dev_dataloader)