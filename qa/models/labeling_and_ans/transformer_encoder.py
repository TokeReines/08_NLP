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
        self.embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        if self.finetune:
            self.dropout = nn.Dropout(encoder_dropout)

        layer = TransformerEncoderLayer(n_heads=8, n_model=self.embed.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer, n_layers=6, n_model=self.embed.n_out)

        self.encoder_n_hidden = self.embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, data, mask, answerable=None, ans_label=None):
        x = self.embed(data)
        x = x if not self.finetune else self.dropout(x)
        x_mask = data.ne(self.pad_index).any(-1)
        x = self.transformer_encoder(x, x_mask)
        
        pred_label = self.labeler(x)

        x_mean = x.mean(dim=1)
        pred_answerable = self.classification(x_mean)

        if answerable is not None and ans_label is not None:
            loss_answerable = self.crit(pred_answerable, answerable)
            d_mask = mask.eq(2)
            pred_label = pred_label[d_mask]
            ans_label = ans_label[ans_label.ge(0)]
            loss_label = self.crit(pred_label, ans_label)
            return loss_answerable + loss_label, pred_label, pred_answerable
        
        return pred_label, pred_answerable

            
    def decode(self, x):
        y = torch.argmax(x, dim=-1)
        return y