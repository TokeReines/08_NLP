import torch
import torch.nn as nn
from modules.attention import DualAttention, DualAttentionLayer, TransformerDualAttention, TransformerDualAttentionLayer
from modules.mlp import MLP
from modules.labeling import Labeler
from models.basemodel import BaseModel

class DualAttentionLabelingAnswerable(BaseModel):
    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, **kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d, **kwargs)

        layer = DualAttentionLayer(self.n_out, att_dropout=0.1)
        self.dual_attention = DualAttention(layer, self.n_out, n_layers=1)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        att_x, att_y = None, None
        if self.return_att:
            x_q, x_d, att_x, att_y = self.dual_attention(x_q, x_d, q_mask, d_mask, self.return_att)
        else:
            x_q, x_d = self.dual_attention(x_q, x_d, q_mask, d_mask, self.return_att)
        return x_q, x_d, att_x, att_y
    
class TransformerDualAttentionLabelingAnswerable(BaseModel):
    def __init__(self, embed_type, name, n_layers=1, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, **kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d, **kwargs)

        layer = TransformerDualAttentionLayer(self.n_out, att_dropout=0.1)
        self.dual_attention = TransformerDualAttention(layer, self.n_out, n_layers=1)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        att_x, att_y = None, None
        if self.return_att:
            x_q, x_d, att_x, att_y = self.dual_attention(x_q, x_d, q_mask, d_mask, self.return_att)
        else:
            x_q, x_d = self.dual_attention(x_q, x_d, q_mask, d_mask, self.return_att)
        return x_q, x_d, att_x, att_y
    


# class TransformerDualAttentionLabelingAnswerable(nn.Module):
#     def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
#         super().__init__()
#         self.pad_index = pad_index
#         self.finetune = finetune
#         self.loss_weights = loss_weights
#         self.ques_embed = TransformerEmbeddingQuesDoc(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
#         # self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
#         if self.finetune:
#             self.dropout = nn.Dropout(encoder_dropout)

#         self.n_out = self.ques_embed.n_out
#         layer = DualAttentionLayer(self.n_out, att_dropout=0.2)
#         self.dual_attention = DualAttention(layer, self.n_out, n_layers=3)

#         self.encoder_n_hidden = self.n_out
#         self.labeler = Labeler(self.encoder_n_hidden)
#         self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
#         self.crit = nn.CrossEntropyLoss()
    
#     def forward(self, ques, doc, answerable=None, ans_label=None, ques_word=None, doc_word=None, start=None, end=None):
#         x_q, x_d = self.ques_embed(ques, doc)
#         # x_d = self.doc_embed(doc)
#         q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
#         m_len = q_mask.shape[1]
        
#         x_q, x_d = self.dual_attention(x_q, x_d, q_mask, d_mask)

#         # x_mean = x.mean(dim=1)
#         lens = torch.sum(d_mask, -1, keepdim=True)
#         x_mean = torch.sum(x_d * d_mask.unsqueeze(-1), dim=1) / lens
#         pred_answerable = self.classification(x_mean)
#         pred_start, pred_end = self.labeler(x_d, d_mask, pred_answerable)

#         if answerable is not None and ans_label is not None:
#             loss_answerable = self.crit(pred_answerable, answerable)
#             loss_label = self.crit(pred_start, start) + self.crit(pred_end, end)
#             loss = loss_answerable + loss_label if self.loss_weights is None else loss_answerable * self.loss_weights + (1 - self.loss_weights) * loss_label
#             return loss, (pred_start, pred_end), pred_answerable
        
#         return (pred_start, pred_end), pred_answerable 
                  
#     def decode(self, x):
#         y = torch.argmax(x, dim=-1)
#         return y
    
#     def decode_qa(self, x, y):
#         b, l = x.shape
#         t = x.new_zeros(*x.shape)
#         x = torch.argmax(x, -1)
#         y = torch.argmax(y, -1)
#         for i in range(b):
#             s = x[i]
#             e = y[i]
#             if s == 0 or e == 0 or s > e:
#                 continue
#             t[i, s:e+1] = 1
#         return t