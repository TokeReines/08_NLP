import pandas as pd
import torch
from models.bert_ans.model import BertAnswerable
from models.labeling_and_ans.transformer_decoder import TransformerDecoderLabelingAnswerable, TransformerEncoderDecoderLabelingAnswerable
from models.labeling_and_ans.transformer_encoder import TransformerEncoderLabelingAnswerable
from models.labeling_and_ans.dualattention import DualAttentionLabelingAnswerable, TransformerDualAttentionLabelingAnswerable
import pandas as pd
from utils.transform import BertTransform, WordTransform
from utils.tokenizer import TransformerTokenizer
from utils.data import BertDataset
from torch.utils.data import DataLoader
from trainer import Trainer, SeqLabelingAnsTrainer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from normalizer import normalize

def visualize(language, exp_path, bert_name, epoch, use_word=False, batch_size=8, embed_type='one-bert', finetune=True, id=5):
    train_set = pd.read_feather(f'./data/{language}_train.feather').iloc[id:id+1]
    shuffle = False
    word_transform = None
    tokens_d = ['<bos>'] + list(train_set['document_plaintext'].iloc[0]) + ['<eos>']
    tokens_q = ['<bos>'] + list(train_set['question_text'].iloc[0]) + ['<eos>']
    print(tokens_q)
    print(tokens_d)
    print(train_set['answer_text'].iloc[0])
    # print(tokens_d[35])

    tokenizer = TransformerTokenizer(bert_name)
    data_transform = BertTransform(tokenizer, fix_len=3, lower_case=True)

    train_dataset = BertDataset(data_transform, 0, train_set, tokenizer, word_transform)

    torch.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.fn_collate, shuffle=shuffle)

    print(len(train_dataloader))

    model = TransformerDualAttentionLabelingAnswerable(embed_type, bert_name, 3, 'mean', tokenizer.vocab[tokenizer.pad], 0.1, 0.2, dropout=0.5, loss_weights=0.2, finetune=finetune, stride=256, return_att=True)

    trainer = SeqLabelingAnsTrainer(f'{exp_path}.pt', model, tokenizer, tokenizer.vocab, 'cuda:0', clip=5, update_steps=1, clip_all=True)
    trainer.load_model(trainer.fname)

    loss, acc, f1 = trainer.evaluate(train_dataloader)
    print(loss, acc, f1)
    y, _, att_x, att_y = trainer.infer(train_dataloader)
    # print(torch.argmax(y[0]).cpu().numpy(), torch.argmax(y[1]).cpu().numpy())
    att_x = att_x.squeeze(0).cpu().numpy()
    att_y = att_y.squeeze(0).cpu().numpy()
    # print(att_y.shape)
    
    
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(att_y, xticklabels=tokens_d, yticklabels=tokens_q, cmap='Blues', vmin=0, vmax=1, linewidths=.5, square=True)
    plt.title('Indonesian Dual Attention Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('./indo_heatmap.pdf', bbox_inches="tight")
    plt.show()

    

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

# visualize('bengali', './exp/label_ans/one_bert/xlm_bn_transdual_finetune', 'xlm-roberta-base', 10, batch_size=4, embed_type='one-bert', finetune=True)
# visualize('arabic', './exp/label_ans/one_bert/xlm_bn_transdual_finetune', 'xlm-roberta-base', 10, batch_size=4, embed_type='one-bert', finetune=True)
visualize('indonesian', './exp/label_ans/one_bert/xlm_id_transdual_finetune', 'xlm-roberta-base', 1, batch_size=1, embed_type='one-bert', id=40)

# training('bengali', './exp/label_ans/one_bert/muril_bn_transdual_finetune', 'google/muril-base-cased', 10, batch_size=4, embed_type='one-bert', finetune=True)