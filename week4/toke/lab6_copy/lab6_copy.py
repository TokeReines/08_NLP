from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from functools import partial
import random
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_metric

import torch_directml
from week4.toke.lab6_copy.lab6_helper import collate_fn, get_train_features, get_validation_features, post_process_predictions, predict, val_collate_fn
from week4.toke.lab6_copy.lab6_train import train

lr = 2e-5
n_epochs = 3
weight_decay = 0.01
warmup_steps = 200
# MODEL_NAME = 'xlm-roberta-base'
dml = torch_directml.device()
device = dml # 'cpu'
MODEL_NAME = 'bert-base-uncased'
language = 'english'

compute_squad = load_metric("squad_v2")
dataset = load_dataset("copenlu/answerable_tydiqa").filter(lambda row: row['language'] == language)

for split in dataset.keys():
    dataset[split] = dataset[split].add_column('id', list(range(len(dataset[split]))))

tk = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_dataset = dataset['train'].map(partial(get_train_features, tk), batched=True, remove_columns=dataset['train'].column_names)
validation_dataset = dataset['validation'].map(partial(get_validation_features, tk), batched=True, remove_columns=dataset['validation'].column_names)

samples = random.sample(list(range(len(tokenized_dataset))), len(dataset['train']))
tokenized_dataset = tokenized_dataset.select(samples)
train_dl = DataLoader(tokenized_dataset, collate_fn=collate_fn, shuffle=True, batch_size=4)


model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps,
    n_epochs * len(train_dl)
)

losses = train(
    model,
    train_dl,
    optimizer,
    scheduler,
    n_epochs,
    device
)

val_dl = DataLoader(validation_dataset,
                    collate_fn=val_collate_fn, batch_size=32)
logits = predict(model, val_dl, device)
predictions = post_process_predictions(
    dataset['validation'], validation_dataset, logits)
formatted_predictions = [{'id': k, 'prediction_text': v,
                          'no_answer_probability': 0.} for k, v in predictions.items()]
gold = [{
    'id': example['id'],
    'answers': {
        'text': example['annotations']['answer_text'],
        'answer_start': example['annotations']['answer_start']}
    }
    for example in dataset['validation']]


print(compute_squad.compute(references=gold, predictions=formatted_predictions))
