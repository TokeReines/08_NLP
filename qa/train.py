import pandas as pd
from models.bert_ans.model import BertAnswerable
from models.labeling_and_ans.transformer_decoder import TransformerDecoderLabelingAnswerable
import pandas as pd
from utils.transform import BertTransform
from utils.tokenizer import TransformerTokenizer
from utils.data import BertDataset
from torch.utils.data import DataLoader
from trainer import Trainer, SeqLabelingAnsTrainer

def training(language, exp_path, bert_name, epoch):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')

    # bert_name = 'aubmindlab/bert-base-arabertv2'
    # bert_name = 'sagorsarker/bangla-bert-base'
    # bert_name = 'cahya/bert-base-indonesian-522M'
    batch_size = 8
    shuffle = True


    optimizer = {
        'name': 'Adam',
        'lr': 1e-4,
        'mu': 0.9,
        'nu': 0.98,
        'eps': 1e-9,
        'weight_decay': 0,
        'lr_rate': 5
    }

    scheduler = {
        'name': 'inverse',
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

    # model = BertAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    model = TransformerDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    # trainer = Trainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0')
    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=5, update_steps=1)
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

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=5, update_steps=1)
    trainer.load_model(trainer.fname)
    loss, acc, f1, _f1 = trainer.evaluate(train_dataloader)
    print(loss, acc, f1, _f1)
    loss, acc, f1, _f1 = trainer.evaluate(dev_dataloader)
    print(loss, acc, f1, _f1)
    loss, acc, f1, _f1 = trainer.evaluate(test_dataloader)
    print(loss, acc, f1, _f1)

training('bengali', './exp/label_ans/bengali_mbert_transformerdecoder_2', 'bert-base-multilingual-cased', 100)
# testing('indonesian', './exp/label_ans/indonesian_bert_transformerdecoder', 'cahya/bert-base-indonesian-522M')