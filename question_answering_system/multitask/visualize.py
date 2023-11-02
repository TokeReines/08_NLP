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
    att_x = att_x.squeeze(0).cpu().numpy()
    att_y = att_y.squeeze(0).cpu().numpy()
    
    
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(att_y, xticklabels=tokens_d, yticklabels=tokens_q, cmap='Blues', vmin=0, vmax=1, linewidths=.5, square=True)
    plt.title('Indonesian Dual Attention Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('./indo_heatmap.pdf', bbox_inches="tight")
    plt.show()
