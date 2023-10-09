import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding
from modules.transformer import TransformerEncoder, TransformerEncoderLayer
from modules.mlp import MLP

class TransformerDecoderLabelingAnswerable(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False):
        super().__init__()
        self.pad_index = pad_index
        self.finetune = finetune
        self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        if self.finetune:
            self.dropout = nn.Dropout(encoder_dropout)

        layer = TransformerEncoderLayer(n_heads=8, n_model=self.embed.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer, n_layers=3, n_model=self.embed.n_out)

        self.encoder_n_hidden = self.embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, ques, doc, answerable=None, ans_label=None, ques_word=None, doc_word=None):
        x_q = self.ques_embed(ques)
        x_d = self.doc_embed(doc)
        q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        m_len = q_mask.shape[1]

        x = torch.cat([x_q, x_d], dim=1)
        x_mask = torch.cat([q_mask, d_mask], dim=1)
        
        x = self.transformer_encoder(x, x_mask)

        pred_label = self.labeler(x[:,m_len:,:])

        # x_mean = x.mean(dim=1)
        lens = torch.sum(x_mask, -1, keepdim=True)
        x_mean = torch.sum(x * x_mask.unsqueeze(-1), dim=1) / lens
        pred_answerable = self.classification(x_mean)

        if answerable is not None and ans_label is not None:
            loss_answerable = self.crit(pred_answerable, answerable)
            m = d_mask & ans_label.ge(0)
            loss_label = self.crit(pred_label[m], ans_label[m])
            loss = loss_answerable + loss_label if self.loss_weights is None else loss_answerable * self.loss_weights + (1 - self.loss_weights) * loss_label
            return loss, pred_label, pred_answerable
        
        return pred_label, pred_answerable
                  
    def decode(self, x):
        y = torch.argmax(x, dim=-1)
        return y