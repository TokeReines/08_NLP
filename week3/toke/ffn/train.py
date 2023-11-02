
import torch_directml
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from week3.toke.ffn.dataset_bow import DatasetBOW
from week3.toke.ffn.dataset import Dataset
from week3.toke.ffn.ffn import FeedForwardNetwork
from week3.toke.ffn.logistic_regression import LogisticRegression
from week3.toke.ffn.tokenizer_bow import TokenizerBOW
from week3.toke.ffn.tokenizer_tfidf import TokenizerTFIDF
from week3.toke.ffn.transform import Transform
from week3.toke.ffn.transform_tfidf import TransformTFIDF
from week3.toke.ffn.trainer import Trainer


train_set = pd.read_feather('week3/toke/data/arabic_train.feather')
dev_set = pd.read_feather('week3/toke/data/arabic_dev.feather')
test_set = pd.read_feather('week3/toke/data/arabic_test.feather')

max_length = 40000
batch_size = 32
hidden_dim = 500

# Example usage
tokenizer = TokenizerTFIDF(train_set, max_length)
transform = TransformTFIDF(tokenizer)

train_dataset = Dataset(train_set, transform, tokenizer)
dev_dataset = Dataset(dev_set, transform, tokenizer)
test_dataset = Dataset(test_set, transform, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LogisticRegression(max_length, 1)
# model = FeedForwardNetwork(max_length, hidden_dim, 1)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dml = torch_directml.device()
trainer = Trainer('week3/toke/models/ffn.pt', model, dml, tokenizer)
trainer.train(optimizer, 50, train_dataloader, dev_dataloader)