import pandas as pd
from sklearn.model_selection import train_test_split
from models.bert_ans.model import BertAnswerable
import pandas as pd
from datasets import load_dataset
from utils.transform import BertTransform
from utils.tokenizer import TransformerTokenizer
from utils.data import BertDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from utils.fn import to_list_tuple
from utils.preprocess import get_tokens

# dataset = load_dataset("copenlu/answerable_tydiqa")
# test_set = dataset["validation"].filter(lambda example, idx: example['language'] == 'arabic', with_indices=True)

# train_set = dataset["train"].filter(lambda example, idx: example['language'] == 'arabic', with_indices=True).to_pandas()
# train_set, dev_set = train_test_split(train_set, test_size=0.15, random_state=42, shuffle=True)
# test_set = test_set.to_pandas()

# l_name = ['question_text', 'answer_start', 'answer_text', 'answerable', 'document_plaintext']

# train_set = to_list_tuple(train_set, l_name)
# dev_set = to_list_tuple(dev_set, l_name)
# test_set = to_list_tuple(test_set, l_name)

# train_set.to_feather('./arabic_train.feather')
# dev_set.to_feather('./arabic_dev.feather')
# test_set.to_feather('./arabic_test.feather')


train_set = pd.read_feather('arabic_train.feather')
dev_set = pd.read_feather('arabic_dev.feather')
test_set = pd.read_feather('arabic_test.feather')

bert_name = 'aubmindlab/bert-base-arabertv2'
batch_size = 16
shuffle = True


optimizer = {
    'name': 'AdamW',
    'lr': 5e-5,
    'mu': 0.9,
    'nu': 0.999,
    'eps': 1e-8,
    'weight_decay': 0,
    'lr_rate': 5
}

scheduler = {
    'name': 'linear',
    'warmup_steps': 10,
    'warmup': 0.1,
    'decay': 0.75,
    'decay_step': 5000
}

tokenizer = TransformerTokenizer(bert_name)
data_transform = BertTransform(tokenizer, fix_len=0)

train_dataset = BertDataset(data_transform, 0, train_set, tokenizer)
dev_dataset = BertDataset(data_transform, 0, dev_set, tokenizer)
test_dataset = BertDataset(data_transform, 0, test_set, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.fn_collate)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=dev_dataset.fn_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.fn_collate)

print(len(train_dataloader))
print(len(dev_dataloader))
print(len(test_dataloader))


model = BertAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
trainer = Trainer("./exp/arabic_bert_answerable_meanpooling.pt", model, tokenizer, tokenizer.vocab, 'cuda:0')
epoch_losses = trainer.train(optimizer, scheduler, 5, train_dataloader, dev_dataloader, test_dataloader)
print(epoch_losses)

