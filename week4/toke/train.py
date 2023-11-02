import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from week4.toke.dataset_bert import DatasetBERT
from week4.toke.spanBERT import SpanBERT
from week4.toke.tokenizer_bert import TokenizerBERT
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from datasets import load_dataset
from week4.toke.trainer_bert import TrainerBERT
import torch_directml
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModel
from torch.utils.data import Subset

max_seq_length = 400
batch_size = 16
lr = 1e-5
max_lr = lr
base_lr=lr
step_size=2000
n_epoch=5
#model_name="aubmindlab/bert-base-arabertv2"
model_name="bert-base-uncased"
# language = 'arabic'
language = 'english'

tokenizer = TokenizerBERT(model_name=model_name, max_seq_length=max_seq_length)

# transform = TransformBOW(tokenizer, max_length=max_length)
dataset = load_dataset("copenlu/answerable_tydiqa")

filtered_dataset = dataset.filter(lambda row:
    row['language'] == language)

# Split the indices in a stratified way
train_set = filtered_dataset["train"]
validation_set = filtered_dataset["validation"]

train_dataset = DatasetBERT(train_set, tokenizer)
val_dataset = DatasetBERT(validation_set, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

bert = BertForQuestionAnswering.from_pretrained(model_name)
model = SpanBERT(bert, tokenizer) 

optimizer = Adam(model.parameters(), lr=lr)
# Use linear scheduler for BERT
scheduler = LinearLR(optimizer, start_factor=1./3., end_factor=1.0, total_iters=len(train_dataloader) * n_epoch)

dml = torch_directml.device()
trainer = TrainerBERT('week3/toke/models/ffn.pt', model, dml, optimizer, scheduler, tokenizer)
trainer.train(n_epoch, train_dataloader, val_dataloader)
