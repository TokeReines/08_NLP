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
# from normalizer import normalize

def training(language, exp_path, bert_name, epoch, use_word=False, batch_size=8, embed_type='one-bert', finetune=True, n_layers_trans=1, lower=True, model=None):
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

    # model = BertAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5)
    # model = TransformerEncoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None)
    # model = DualAttentionLabelingAnswerable(embed_type, bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.5, finetune=finetune, stride=256)
    # model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256)
    # model = TransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256, n_layers_trans=n_layers_trans)
    # model = TransformerDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256)
    # model = TransformerDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256)
    # model = TransformerEncoderDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256)
    # model = TransformerDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.3)
    # model = TransformerEncoderDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.3)
    # trainer = Trainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0')
    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0.5, update_steps=1, clip_all=True)
    epoch_losses = trainer.train(optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader)
    print(epoch_losses)
    # trainer.load_model(trainer.fname)
    # loss, acc, f1 = trainer.evaluate(train_dataloader)
    # print(loss, acc, f1)
    # loss, acc, f1 = trainer.evaluate(dev_dataloader)
    # print(loss, acc, f1)
    # loss, acc, f1 = trainer.evaluate(test_dataloader)
    # print(loss, acc, f1)

def testing(language, exp_path, bert_name, model, lower=False):
    # train_set = pd.read_feather(f'./data/{language}_train.feather')
    # dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')
    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=0, lower_case=lower)

    # train_dataset = BertDataset(data_transform, 0, train_set, tokenizer)
    # dev_dataset = BertDataset(data_transform, 0, dev_set, tokenizer)
    test_dataset = BertDataset(data_transform, 0, test_set, tokenizer)
    batch_size = 8

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.fn_collate)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=dev_dataset.fn_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.fn_collate)
    # model = TransformerDecoderLabelingAnswerable(bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None)

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=0, update_steps=2)
    trainer.load_model(trainer.fname)
    # loss, acc, f1 = trainer.evaluate(train_dataloader)
    # print(loss, acc, f1)
    # loss, acc, f1 = trainer.evaluate(dev_dataloader)
    # print(loss, acc, f1)
    print(language, exp_path)
    loss, acc, f1 = trainer.evaluate(test_dataloader)
    print(loss, acc, f1)

embed_type = 'one-bert'
bert_name = 'google/muril-base-cased'

tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256, finetune=True)
# training('indonesian', './exp/label_ans/one_bert/xlm_id_encoder_finetune', bert_name=bert_name, epoch=5, batch_size=1, model=model)
# testing('indonesian', './exp/label_ans/one_bert/xlm_id_encoder_finetune', bert_name, model)

# model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 4, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=24)
# training('bengali', './exp/label_ans/one_bert/test3', bert_name=bert_name, epoch=5, batch_size=4, model=model)
# testing('bengali', './exp/label_ans/one_bert/test3', bert_name, model)

# model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 4, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=48)
# training('bengali', './exp/label_ans/one_bert/test4', bert_name=bert_name, epoch=5, batch_size=4, model=model)
# testing('bengali', './exp/label_ans/one_bert/test4', bert_name, model)

# model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=64)
# training('bengali', './exp/label_ans/one_bert/test5', bert_name=bert_name, epoch=5, batch_size=4, model=model)
# testing('bengali', './exp/label_ans/one_bert/test5', bert_name, model)

# model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=128)
# training('bengali', './exp/label_ans/one_bert/test6', bert_name=bert_name, epoch=5, batch_size=4, model=model)
# testing('bengali', './exp/label_ans/one_bert/test6', bert_name, model)

model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=12)
training('bengali', './exp/label_ans/one_bert/test11', bert_name=bert_name, epoch=5, batch_size=4, model=model)
testing('bengali', './exp/label_ans/one_bert/test11', bert_name, model)

# model = ConvTransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=None, stride=256, finetune=True, conv_stride=9)
# training('bengali', './exp/label_ans/one_bert/test10', bert_name=bert_name, epoch=5, batch_size=4, model=model)
# testing('bengali', './exp/label_ans/one_bert/test10', bert_name, model)

# model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256, finetune=True)
# training('indonesian', './exp/label_ans/one_bert/xlm_id_transdual_finetune', bert_name=bert_name, epoch=5, batch_size=1, model=model)
# testing('indonesian', './exp/label_ans/one_bert/xlm_id_transdual_finetune', bert_name, model)

# training('bengali', './exp/label_ans/bengali_bert_transformerdecoder_start_end_pred', 'sagorsarker/bangla-bert-base', 25, batch_size=4)
# training('bengali', './exp/label_ans/test_bn_banglabert', 'sagorsarker/bangla-bert-base', 30, batch_size=12)
# training('bengali', './exp/label_ans/test_bn_muril_2', 'google/muril-base-cased', 30, batch_size=8)
# training('indonesian', './exp/label_ans/test_id_xlm', 'xlm-roberta-base', 25, batch_size=3)
# training('arabic', './exp/label_ans/test_ar.xlm', 'xlm-roberta-base', 25, batch_size=1)
# training('bengali', './exp/label_ans/test_bn_xlm_4', 'xlm-roberta-base', 30, batch_size=8)
# training('indonesian', './exp/label_ans/indo_bert_transformerencoderdecoder_2', 'cahya/bert-base-indonesian-522M', 100) 
# training('indonesian', './exp/label_ans/indonesian_xlmr_transformerencoder_start_end_pred_3', 'xlm-roberta-base', 30, batch_size=2)
# training('bengali', './exp/label_ans/one_bert/xlm_bn_transdual_nofinetune', 'xlm-roberta-base', 10, batch_size=16, embed_type='one-bert', finetune=False)
# training('arabic', './exp/label_ans/two_bert/xlm_ar_transdual_nofinetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='two-bert', finetune=False)
# training('arabic', './exp/label_ans/one_bert/xlm_ar_transdual_finetune', 'xlm-roberta-base', 5, batch_size=1, embed_type='one-bert', finetune=True)
# training('arabic', './exp/label_ans/one_bert/arabert_ar_transdual_finetune', 'aubmindlab/bert-base-arabertv2', 10, batch_size=1, embed_type='one-bert', finetune=True)
# training('arabic', './exp/label_ans/one_bert/xlm_id_transdual_finetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='one-bert', finetune=True)
# training('bengali', './exp/label_ans/one_bert/xlm_id_transdual_finetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='one-bert', finetune=True)
# training('indonesian', './exp/label_ans/one_bert/xlm_id_transdual_finetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='one-bert', finetune=True)
# training('indonesian', './exp/label_ans/one_bert/bertindo_id_dual_finetune', 'cahya/bert-base-indonesian-522M', 5, batch_size=2, embed_type='one-bert')
# training('arabic', './exp/label_ans/one_bert/arabert_ar_dual_finetune', 'aubmindlab/bert-base-arabertv2', 5, batch_size=2, embed_type='one-bert')
# training('bengali', './exp/label_ans/one_bert/banglabert_bn_dual_finetune', 'sagorsarker/bangla-bert-base', 5, batch_size=4, embed_type='one-bert')
# training('bengali', './exp/label_ans/one_bert/muril_bn_dual_finetune', 'google/muril-base-cased', 5, batch_size=4, embed_type='one-bert')

# training('arabic', './exp/label_ans/one_bert/arabert_ar_decoder_finetune', 'aubmindlab/bert-base-arabertv2', 10, batch_size=1, embed_type='one-bert', finetune=True, n_layers_trans=1)
# training('indonesian', './exp/label_ans/one_bert/xlm_id_decoder_finetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='one-bert', finetune=True, n_layers_trans=2)
# training('bengali', './exp/label_ans/one_bert/muril_bn_decoder_finetune', 'google/muril-base-cased', 10, batch_size=4, embed_type='one-bert', finetune=True, n_layers_trans=2)
# training('arabic', './exp/label_ans/one_bert/xlm_bn_transdual_finetune', 'xlm-roberta-base', 10, batch_size=4, embed_type='one-bert', finetune=True)
# training('indonesian', './exp/label_ans/one_bert/xlm_bn_transdual_finetune', 'xlm-roberta-base', 10, batch_size=4, embed_type='one-bert', finetune=True)
# training('bengali', './exp/label_ans/one_bert/muril_bn_encdec_finetune', 'google/muril-base-cased', 10, batch_size=4, embed_type='one-bert', finetune=True, lower=False)
# training('indonesian', './exp/label_ans/one_bert/xlm_id_encdec_finetune', 'xlm-roberta-base', 10, batch_size=2, embed_type='one-bert', finetune=True, lower=True)

# training('bengali', './exp/label_ans/one_bert/muril_bn_transdual_finetune', 'google/muril-base-cased', 10, batch_size=4, embed_type='one-bert', finetune=True)

# embed_type = 'one-bert'
# bert_name = 'aubmindlab/bert-base-arabertv2'

# tokenizer = TransformerTokenizer(bert_name)
# model = DualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('arabic', './exp/label_ans/one_bert/arabert_ar_dual_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256, n_layers_trans=1)
# testing('arabic', './exp/label_ans/one_bert/arabert_ar_encoder_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('arabic', './exp/label_ans/one_bert/arabert_ar_transdual_finetune', bert_name, model)

# embed_type = 'one-bert'
# bert_name = 'cahya/bert-base-indonesian-522M'

# tokenizer = TransformerTokenizer(bert_name)
# model = DualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('indonesian', './exp/label_ans/one_bert/bertindo_id_dual_finetune', bert_name, model, lower=True)

# embed_type = 'one-bert'
# bert_name = 'cahya/bert-base-indonesian-1.5G'

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('indonesian', './exp/label_ans/one_bert/idbert_id_decoder_finetune', bert_name, model, lower=True)

# embed_type = 'one-bert'
# bert_name = 'cahya/bert-base-indonesian-1.5G'

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256, n_layers_trans=2)
# testing('indonesian', './exp/label_ans/one_bert/idbert_id_encoder_finetune', bert_name, model, lower=True)

# embed_type = 'one-bert'
# bert_name = 'google/muril-base-cased'

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('bengali', './exp/label_ans/one_bert/muril_bn_decoder_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = DualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('bengali', './exp/label_ans/one_bert/muril_bn_dual_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('bengali', './exp/label_ans/one_bert/muril_bn_encdec_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256, n_layers_trans=2)
# testing('bengali', './exp/label_ans/one_bert/muril_bn_encoder_finetune', bert_name, model)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('bengali', './exp/label_ans/one_bert/muril_bn_transdual_finetune', bert_name, model)

# embed_type = 'one-bert'
# bert_name = 'xlm-roberta-base'

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('indonesian', './exp/label_ans/one_bert/xlm_id_decoder_finetune', bert_name, model, lower=True)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('indonesian', './exp/label_ans/one_bert/xlm_id_transdual_finetune', bert_name, model, lower=True)

# tokenizer = TransformerTokenizer(bert_name)
# model = TransformerEncoderDecoderLabelingAnswerable(embed_type, bert_name, 1, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, stride=256)
# testing('indonesian', './exp/label_ans/one_bert/xlm_id_encdec_finetune', bert_name, model, lower=True)