import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding
from modules.transformer import TransformerEncoder, TransformerEncoderLayer
from modules.mlp import MLP
from modules.labeling import Labeler

class TransformerEncoderLabelingAnswerable(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
        super().__init__()
        self.pad_index = pad_index
        self.finetune = finetune
        self.loss_weights = loss_weights
        self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=512)
        self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=512)
        if self.finetune:
            self.dropout = nn.Dropout(encoder_dropout)

        self.n_out = self.ques_embed.n_out
        layer = TransformerEncoderLayer(n_heads=8, n_model=self.n_out, n_inner=768)
        self.transformer_encoder = TransformerEncoder(layer, n_layers=5, n_model=self.n_out)

        self.encoder_n_hidden = self.n_out
        # self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.labeler = Labeler(self.encoder_n_hidden)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, ques, doc, answerable=None, ans_label=None, ques_word=None, doc_word=None, start=None, end=None):
        x_q = self.ques_embed(ques)
        x_d = self.doc_embed(doc)
        q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        m_len = q_mask.shape[1]

        x = torch.cat([x_q, x_d], dim=1)
        x_mask = torch.cat([q_mask, d_mask], dim=1)
        
        x = self.transformer_encoder(x, x_mask)

        # pred_label = self.labeler(x[:,m_len:,:])
        pred_start, pred_end = self.labeler(x[:,m_len:,:], d_mask)

        # x_mean = x.mean(dim=1)
        lens = torch.sum(x_mask, -1, keepdim=True)
        x_mean = torch.sum(x * x_mask.unsqueeze(-1), dim=1) / lens
        pred_answerable = self.classification(x_mean)

        if answerable is not None and ans_label is not None:
            loss_answerable = self.crit(pred_answerable, answerable)
            # m = d_mask & ans_label.ge(0)
            # loss_label = self.crit(pred_label[m], ans_label[m])
            # print(m.shape, pred_start.shape, pred_end.shape, start, end)
            loss_label = self.crit(pred_start, start) + self.crit(pred_end, end)
            loss = loss_answerable + loss_label if self.loss_weights is None else loss_answerable * self.loss_weights + (1 - self.loss_weights) * loss_label
            return loss, (pred_start, pred_end), pred_answerable
        
        return (pred_start, pred_end), pred_answerable
                  
    def decode(self, x):
        y = torch.argmax(x, dim=-1)
        return y
    
    def decode_qa(self, x, y):
        b, l = x.shape
        t = x.new_zeros(*x.shape)
        x = torch.argmax(x, -1)
        y = torch.argmax(y, -1)
        for i in range(b):
            s = x[i]
            e = y[i]
            if s == 0 or e == 0 or s > e:
                continue
            t[i, s:e+1] = 1
        return t
