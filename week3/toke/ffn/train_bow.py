
import torch_directml
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from week3.toke.ffn.dataset_bow import DatasetBOW
from week3.toke.ffn.ffn import FeedForwardNetwork
from week3.toke.ffn.tokenizer_bow import TokenizerBOW
from week3.toke.ffn.transform import TransformBOW
from week3.toke.trainer import Trainer


train_set = pd.read_feather('week3/toke/data/arabic_train.feather')
dev_set = pd.read_feather('week3/toke/data/arabic_dev.feather')
test_set = pd.read_feather('week3/toke/data/arabic_test.feather')

max_length = 100000
# Example usage
tokenizer = TokenizerBOW(train_set, max_length)
transform = TransformBOW(tokenizer)

train_dataset = DatasetBOW(train_set, transform, tokenizer)
dev_dataset = DatasetBOW(dev_set, transform, tokenizer)
test_dataset = DatasetBOW(test_set, transform, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

hidden_dim = 500
model = FeedForwardNetwork(max_length*2+1, hidden_dim, 1)
optimizer = optim.SGD(model.parameters(), lr=0.000001)

dml = torch_directml.device()
trainer = Trainer('week3/toke/models/ffn.pt', model, dml, tokenizer)
trainer.train(optimizer, 50, train_dataloader, dev_dataloader)