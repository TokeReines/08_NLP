import pandas as pd
import torch
from models.bert_ans.model import BertAnswerable
from models.labeling_and_ans.transformer_decoder import TransformerDecoderLabelingAnswerable, TransformerEncoderDecoderLabelingAnswerable
from models.labeling_and_ans.transformer_encoder import TransformerEncoderLabelingAnswerable
from models.labeling_and_ans.dualattention import DualAttentionLabelingAnswerable, TransformerDualAttentionLabelingAnswerable
from models.labeling_and_ans.poolingattention import ConvTransformerEncoderLabelingAnswerable
import pandas as pd
from utils.transform import BertTransform, WordTransform
from utils.tokenizer import TransformerTokenizer
from utils.data import BertDataset
from torch.utils.data import DataLoader
from trainer import Trainer, SeqLabelingAnsTrainer

def training(language, exp_path, bert_name, epoch, use_word=False, batch_size=8, embed_type='one-bert', finetune=True, n_layers_trans=1, lower=True, model=None):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')

    shuffle = True
    word_transform = None
    if use_word:
        data = [list(train_set["document_plaintext"]), list(train_set["question_text"])]
        word_transform = WordTransform(data, lower_case=False)
        n_vocab = len(word_transform.vocab)
        word_pad_index = word_transform.vocab['<pad>']
        ques_max_len = 100
        doc_max_len = 5000

    optimizer = {
        'name': 'AdamW',
        'lr': 2e-5,
        'mu': 0.9,
        'nu': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.01,
        'lr_rate': 10
    }

    scheduler = {
        'name': 'linear',
        'warmup_steps': 200,
        'warmup': 0.1,
        'decay': 0.75,
        'decay_step': 5000
    }

    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=0, lower_case=lower)

    train_dataset = BertDataset(data_transform, 0, train_set, tokenizer, word_transform)
    dev_dataset = BertDataset(data_transform, 0, dev_set, tokenizer, word_transform)
    test_dataset = BertDataset(data_transform, 0, test_set, tokenizer, word_transform)
    torch.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.fn_collate, shuffle=shuffle)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=dev_dataset.fn_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.fn_collate)

    print(len(train_dataloader))
    print(len(dev_dataloader))
    print(len(test_dataloader))

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0.5, update_steps=1, clip_all=True)
    epoch_losses = trainer.train(optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader)
    print(epoch_losses)

def testing(language, exp_path, bert_name, model, lower=False):
    test_set = pd.read_feather(f'./data/{language}_test.feather')
    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=0, lower_case=lower)

    test_dataset = BertDataset(data_transform, 0, test_set, tokenizer)
    batch_size = 8

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.fn_collate)

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0, update_steps=2)
    trainer.load_model(trainer.fname)
    print(language, exp_path)
    loss, acc, f1 = trainer.evaluate(test_dataloader)
    print(loss, acc, f1)

embed_type = 'one-bert'
bert_name = 'google/muril-base-cased'
tokenizer = TransformerTokenizer(bert_name)
model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=12)
training('bengali', './exp/label_ans/one_bert/test11', bert_name=bert_name, epoch=5, batch_size=4, model=model)
testing('bengali', './exp/label_ans/one_bert/test11', bert_name, model)