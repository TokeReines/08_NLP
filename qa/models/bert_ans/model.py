import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding
from modules.mlp import MLP

class BertAnswerable(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0.):
        super().__init__()
        self.pad_index = pad_index

        self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=True, stride=128)
        self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=False, stride=128)
        self.ques_encoder_dropout = nn.Dropout(p=encoder_dropout)
        self.doc_encoder_dropout = nn.Dropout(p=encoder_dropout)
        self.encoder_n_hidden = self.ques_embed.n_out + self.doc_embed.n_out
        self.projection = MLP(self.encoder_n_hidden, 256, dropout)
        self.classification = nn.Linear(256, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, ques, doc, label):
        x_q = self.ques_embed(ques)
        x_d = self.doc_embed(doc)
        x_q = self.ques_encoder_dropout(x_q)
        x_d = self.doc_encoder_dropout(x_d)
        x_q, x_d = x_q[:, 0, :], x_d[:, 0, :]
        # x_q = x_q.mean(dim=1)
        # x_d = x_d.mean(dim=1)
        x = torch.cat([x_q, x_d], dim=1)
        x = self.projection(x)
        x = self.classification(x)
        if label is not None:
            loss = self.crit(x, label)
            return loss, x
        return x
    
    def decode(self, x):
        y = torch.argmax(x, dim=1)
        return y
    

