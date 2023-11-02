from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from datasets import load_dataset
from week4.toke.dataset_bert import DatasetBERT
from week4.toke.spanBERT import SpanBERT

from week4.toke.tokenizer_bert import TokenizerBERT
from week4.toke.trainer_bert import TrainerBERT

max_seq_length = 512
batch_size = 16
lr = 1e-5
max_lr = lr
base_lr=lr
step_size=2000
n_epoch=50
#model_name="aubmindlab/bert-base-arabertv2"
model_name="bert-base-uncased"
# language = 'arabic'
language = 'english'

tokenizer = TokenizerBERT(model_name=model_name, max_seq_length=max_seq_length)

# transform = TransformBOW(tokenizer, max_length=max_length)
dataset = load_dataset("copenlu/answerable_tydiqa")

dataset = dataset.filter(lambda row: row['language'] == language and len(row['document_plaintext']) < max_seq_length and len(row['question_text']) < max_seq_length)

train_set = dataset["train"][:100]
validation_set = dataset["validation"][:100]
train_dataset = DatasetBERT(train_set, tokenizer)
val_dataset = DatasetBERT(validation_set, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

bert = BertForQuestionAnswering.from_pretrained(model_name)
model = SpanBERT(bert, tokenizer)

# optimizer = Adam(model.parameters(), lr=lr)
# scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=1, step_size_down=len(train_dataloader) * n_epoch)

optimizer = Adam(model.parameters(), lr=lr)
# scheduler = CyclicLR(optimizer, base_lr=0., max_lr=lr, step_size_up=1, step_size_down=len(train_dataloader)*n_epoch, cycle_momentum=False)
# Use linear scheduler for BERT
scheduler = LinearLR(optimizer, start_factor=1./3., end_factor=1.0, total_iters=len(train_dataloader) * n_epoch)

dml = 'cpu' # torch_directml.device()
trainer = TrainerBERT('week3/toke/models/ffn.pt', model, dml, optimizer, scheduler, )
trainer.train(n_epoch, train_dataloader, val_dataloader)
