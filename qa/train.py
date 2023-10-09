import pandas as pd
import torch
from models.bert_ans.model import BertAnswerable
from models.labeling_and_ans.transformer_decoder import TransformerDecoderLabelingAnswerable, TransformerEncoderDecoderLabelingAnswerable
from models.labeling_and_ans.transformer_encoder import TransformerEncoderLabelingAnswerable
import pandas as pd
from utils.transform import BertTransform, WordTransform
from utils.tokenizer import TransformerTokenizer
from utils.data import BertDataset
from torch.utils.data import DataLoader
from trainer import Trainer, SeqLabelingAnsTrainer
from normalizer import normalize

def training(language, exp_path, bert_name, epoch, use_word=False, batch_size=8):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')

    # bert_name = 'aubmindlab/bert-base-arabertv2'
    # bert_name = 'sagorsarker/bangla-bert-base'
    # bert_name = 'cahya/bert-base-indonesian-522M'
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
        'name': 'Adam',
        'lr': 5e-5,
        'mu': 0.9,
        'nu': 0.999,
        'eps': 1e-8,
        'weight_decay': 0,
        'lr_rate': 10
    }

    scheduler = {
        'name': 'linear',
        'warmup_steps': 10,
        'warmup': 0.1,
        'decay': 0.75,
        'decay_step': 5000
    }

    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=0, lower_case=False)

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

    # model = BertAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    model = TransformerEncoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    # model = TransformerDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    # model = TransformerEncoderDecoderLabelingAnswerable(bert_name, 2, pooling='mean', pad_index=tokenizer.vocab[tokenizer.pad], mix_dropout=0.2, encoder_dropout=0.2, dropout=0.5, finetune=False, stride=128, loss_weights=None, ans_classification=True, concat_ques_doc=False)
    # trainer = Trainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0')
    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0, update_steps=1)
    epoch_losses = trainer.train(optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader)
    print(epoch_losses)

def testing(language, exp_path, bert_name):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')
    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=0)

    train_dataset = BertDataset(data_transform, 0, train_set, tokenizer)
    dev_dataset = BertDataset(data_transform, 0, dev_set, tokenizer)
    test_dataset = BertDataset(data_transform, 0, test_set, tokenizer)
    batch_size = 4

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.fn_collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=dev_dataset.fn_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.fn_collate)
    model = TransformerDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None)

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0, update_steps=1)
    trainer.load_model(trainer.fname)
    loss, acc, f1, _f1 = trainer.evaluate(train_dataloader)
    print(loss, acc, f1, _f1)
    loss, acc, f1, _f1 = trainer.evaluate(dev_dataloader)
    print(loss, acc, f1, _f1)
    loss, acc, f1, _f1 = trainer.evaluate(test_dataloader)
    print(loss, acc, f1, _f1)

# training('bengali', './exp/label_ans/bengali_bert_transformerencoder', 'sagorsarker/bangla-bert-base', 30)
# training('indonesian', './exp/label_ans/indo_bert_transformerencoderdecoder_2', 'cahya/bert-base-indonesian-522M', 100) 
training('indonesian', './exp/label_ans/indonesian_bert_transformerencoder', 'cahya/bert-base-indonesian-522M', 30, batch_size=4)
training('arabic', './exp/label_ans/arabic_bert_transformerencoder', 'aubmindlab/bert-base-arabertv2', 30, batch_size=1)