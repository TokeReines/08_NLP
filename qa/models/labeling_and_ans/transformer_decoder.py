import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.bertembed import TransformerEmbedding
from modules.lstm import VariationalLSTM
from modules.dropout import SharedDropout
from modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, MultilevelEmbedding
from modules.labeling import Labeler
from modules.mlp import MLP
from utils.fn import pad
from models.basemodel import BaseModel

class TransformerDecoderLabelingAnswerable(BaseModel):

    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, **kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d=False, **kwargs)

        layer = TransformerDecoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=2, n_model=self.n_out)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)
        return x_q, x, None, None
    
    def scorer(self, ques, q_mask, doc=None, d_mask=None, m_len=None):
        pred_start, pred_end = self.labeler(doc, d_mask, None)
        lens = torch.sum(d_mask, -1, keepdim=True)
        x_mean = torch.sum(doc * d_mask.unsqueeze(-1), dim=1) / lens
        pred_answerable = self.classification(x_mean)
        return pred_answerable, pred_start, pred_end 
    
class TransformerEncoderDecoderLabelingAnswerable(BaseModel):

    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, **kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d, **kwargs)

        layer_encoder = TransformerEncoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer_encoder, n_layers=1, n_model=self.n_out)
        layer = TransformerDecoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=1, n_model=self.n_out)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        x_q = self.transformer_encoder(x_q, q_mask)
        x_d = self.transformer_decoder(x_d, x_q, d_mask, q_mask)
        return x_q, x_d, None, None
    

# class TransformerDecoderLabelingAnswerable(nn.Module):
#     def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, word=False, n_vocab=0, n_embed=256, word_pad_index=0, ques_max_len=512, doc_max_len=512):
#         super().__init__()
#         self.pad_index = pad_index
#         self.use_word = word

#         if self.use_word:
#             self.ques_word_embed = MultilevelEmbedding(n_vocab, n_embed, max_len=ques_max_len, pad_index=word_pad_index)
#             self.doc_word_embed = MultilevelEmbedding(n_vocab, n_embed, max_len=doc_max_len, pad_index=word_pad_index)

#             self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=n_embed // 2)
#             self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=n_embed // 2)
#             self.n_out = self.word_embed.n_out
#         else:
#             self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
#             self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)

#             self.n_out = self.ques_embed.n_out
#         self.loss_weights = loss_weights


#         layer = TransformerDecoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
#         self.transformer_decoder = TransformerDecoder(layer, n_layers=2, n_model=self.n_out)

#         self.encoder_n_hidden = self.n_out
#         # self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
#         self.labeler = Labeler(self.encoder_n_hidden)
#         self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
#         # self.labeler = MLP(self.ques_embed.n_out, 2, dropout, activation=False)
#         # self.classification = MLP(self.ques_embed.n_out, 2, dropout, activation=False)
#         self.crit = nn.CrossEntropyLoss()
    
#     def forward(self, ques, doc, answerable=None, ans_label=None, ques_word=None, doc_word=None, start=None, end=None):
#         x_q = self.ques_embed(ques)
#         x_d = self.doc_embed(doc)
#         q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
#         if self.use_word:
#             x_q = self.ques_word_embed(ques_word, x_q)
#             x_d = self.doc_word_embed(doc_word, x_d)
        
#         x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)

#         # pred_label = self.labeler(x)
#         pred_start, pred_end = self.labeler(x, d_mask)

#         # x_mean = x.mean(dim=1)
#         lens = torch.sum(d_mask, -1, keepdim=True)
#         x_mean = torch.sum(x * d_mask.unsqueeze(-1), dim=1) / lens
#         pred_answerable = self.classification(x_mean)

#         if answerable is not None and ans_label is not None:
#             loss_answerable = self.crit(pred_answerable, answerable)
#             # m = d_mask & ans_label.ge(0)
#             # loss_label = self.crit(pred_label[m], ans_label[m])
#             loss_label = self.crit(pred_start, start) + self.crit(pred_end, end)
#             loss = loss_answerable + loss_label if self.loss_weights is None else loss_answerable * self.loss_weights + (1 - self.loss_weights) * loss_label
#             return loss, (pred_start, pred_end), pred_answerable
        
#         return loss, (pred_start, pred_end), pred_answerable
            
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


# class TransformerEncoderDecoderLabelingAnswerable(nn.Module):
#     def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, concat_ques_doc=False, use_encoder=True, use_decoder=True, ans_classification=True):
#         super().__init__()
#         self.pad_index = pad_index
#         self.loss_weights = loss_weights
#         self.finetune = finetune
#         self.concat_ques_doc = concat_ques_doc
#         self.use_encoder = use_encoder
#         self.use_decoder = use_decoder
#         self.ans_classification = ans_classification

#         assert use_encoder or use_decoder

#         self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=512)
#         self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=512)
#         if self.finetune:
#             self.embed_dropout = nn.Dropout(encoder_dropout)
#         self.n_out = self.ques_embed.n_out

#         if self.use_encoder:
#             layer_encoder = TransformerEncoderLayer(n_heads=8, n_model=self.n_out, n_inner=768)
#             self.transformer_encoder = TransformerEncoder(layer_encoder, n_layers=3, n_model=self.n_out)
#         if self.use_decoder:
#             layer = TransformerDecoderLayer(n_heads=8, n_model=self.n_out, n_inner=768)
#             self.transformer_decoder = TransformerDecoder(layer, n_layers=3, n_model=self.n_out)

#         self.encoder_n_hidden = self.n_out
#         self.labeler = Labeler(self.encoder_n_hidden)
#         if ans_classification:
#             self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
#         self.crit = nn.CrossEntropyLoss()
    
#     def do_lstm(self, x_q, q_mask, x_d, d_mask):
#         x_q = pack_padded_sequence(x_q, q_mask.sum(1).tolist(), True, False)
#         x_q, _ = self.ques_lstm_encoder(x_q)
#         x_q, _ = pad_packed_sequence(x_q, True, total_length=q_mask.shape[1])

#         x_d = pack_padded_sequence(x_d, d_mask.sum(1).tolist(), True, False)
#         x_d, _ = self.ques_lstm_encoder(x_d)
#         x_d, _ = pad_packed_sequence(x_d, True, total_length=d_mask.shape[1])

#         x_q = self.ques_embed_dropout(x_q)
#         x_d = self.doc_embed_dropout(x_d)

#         return x_q, x_d
    
#     def forward(self, ques, doc, answerable=None, ans_label=None, start=None, end=None):
#         x_q = self.ques_embed(ques)
#         x_d = self.doc_embed(doc)

#         q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
#         m_len = q_mask.shape[1]
#         x_cat = torch.cat([x_q, x_d], dim=1)
#         x_mask = torch.cat([q_mask, d_mask], dim=1)
#         x = self.transformer_encoder(x_cat, x_mask)
#         x = self.transformer_decoder(x_d, x_cat, d_mask, x_mask)
#         pred_start, pred_end = self.labeler(x, d_mask)
#         # print(pred_label)
#         # x_mean = x.mean(dim=1)
#         lens = torch.sum(x_mask, -1, keepdim=True)
#         x_mean = torch.sum(x_cat * x_mask.unsqueeze(-1), dim=1) / lens
#         pred_answerable = self.classification(x_mean)

#         if answerable is not None and ans_label is not None:
#             loss_answerable = self.crit(pred_answerable, answerable)
#             # m = d_mask & ans_label.ge(0)
#             # loss_label = self.crit(pred_label[m], ans_label[m])
#             # print(m.shape, pred_start.shape, pred_end.shape, start, end)
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