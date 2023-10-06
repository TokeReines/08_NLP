import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding
from modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from modules.mlp import MLP
from utils.fn import pad

class TransformerDecoderLabelingAnswerable(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
        super().__init__()
        self.pad_index = pad_index

        self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)

        self.loss_weights = loss_weights

        layer = TransformerDecoderLayer(n_heads=8, n_model=self.ques_embed.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=3, n_model=self.ques_embed.n_out)

        self.encoder_n_hidden = self.ques_embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, ques, doc, answerable=None, ans_label=None):
        x_q = self.ques_embed(ques)
        x_d = self.doc_embed(doc)
        q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)

        pred_label = self.labeler(x)

        x_mean = x.mean(dim=1)
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
    

class TransformerDecoderLabelingAnswerableConcat(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
        super().__init__()
        self.pad_index = pad_index
        self.finetune = finetune
        self.loss_weights = loss_weights
        self.embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        if self.finetune:
            self.embed_dropout = nn.Dropout(encoder_dropout)
            
        layer = TransformerDecoderLayer(n_heads=8, n_model=self.ques_embed.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=3, n_model=self.ques_embed.n_out)

        self.encoder_n_hidden = self.ques_embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, data, mask, answerable=None, ans_label=None):
        x = self.embed(data)
        if self.finetune:
            x = self.embed_dropout(x)

        # x_q = x[mask.eq(1)]
        # x_d = x[mask.eq(2)]
        # q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        # sum_ques_len = mask.eq(1).sum(-1)
        # max_ques_len = sum_ques_len.max()
        # min_ques_len = sum_ques_len.min()

        # x_q, q_mask = x[:, :max_ques_len, :], mask[:, :max_ques_len]
        # x_d, d_mask = x[:, min_ques_len:, :], mask[:, min_ques_len:]
        q_mask, d_mask = mask.eq(1), mask.eq(2)
        q_lens, d_lens = q_mask.sum(-1), d_mask.sum(-1)
        x_q = pad(x[q_mask].split(q_lens.tolist()), 0)
        x_d = pad(x[d_mask].split(d_lens.tolist()), 0)
        q_mask = pad(q_mask[q_mask].split(q_lens.tolist()), False)
        d_mask = pad(d_mask[d_mask].split(d_lens.tolist()), False)

        x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)

        pred_label = self.labeler(x)

        x_mean = x.mean(dim=1)
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
    
class TransformerEncoderDecoderLabelingAnswerableConcat(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
        super().__init__()
        self.pad_index = pad_index
        self.finetune = finetune
        self.loss_weights = loss_weights
        self.embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, n_out=512)
        if self.finetune:
            self.embed_dropout = nn.Dropout(encoder_dropout)

        layer_encoder = TransformerEncoderLayer(n_heads=4, n_model=self.embed.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer_encoder, n_layers=2, n_model=self.embed.n_out)
            
        layer = TransformerDecoderLayer(n_heads=4, n_model=self.embed.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=2, n_model=self.embed.n_out)

        self.encoder_n_hidden = self.embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, data, mask, answerable=None, ans_label=None):
        x = self.embed(data)
        if self.finetune:
            x = self.embed_dropout(x)

        # x_q = x[mask.eq(1)]
        # x_d = x[mask.eq(2)]
        # q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        # sum_ques_len = mask.eq(1).sum(-1)
        # max_ques_len = sum_ques_len.max()
        # min_ques_len = sum_ques_len.min()

        # x_q, q_mask = x[:, :max_ques_len, :], mask[:, :max_ques_len]
        # x_d, d_mask = x[:, min_ques_len:, :], mask[:, min_ques_len:]
        q_mask, d_mask = mask.eq(1), mask.eq(2)
        q_lens, d_lens = q_mask.sum(-1), d_mask.sum(-1)
        x_q = pad(x[q_mask].split(q_lens.tolist()), 0)
        x_d = pad(x[d_mask].split(d_lens.tolist()), 0)
        q_mask = pad(q_mask[q_mask].split(q_lens.tolist()), False)
        d_mask = pad(d_mask[d_mask].split(d_lens.tolist()), False)

        x_q = self.transformer_encoder(x_q, q_mask)

        x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)

        pred_label = self.labeler(x)

        x_mean = x.mean(dim=1)
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
    
    
class TransformerEncoderDecoderLabelingAnswerable(nn.Module):
    def __init__(self, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None):
        super().__init__()
        self.pad_index = pad_index

        self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
        self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)

        self.loss_weights = loss_weights

        layer_encoder = TransformerEncoderLayer(n_heads=8, n_model=self.ques_embed.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer_encoder, n_layers=3, n_model=self.ques_embed.n_out)

        layer = TransformerDecoderLayer(n_heads=8, n_model=self.ques_embed.n_out, n_inner=1024)
        self.transformer_decoder = TransformerDecoder(layer, n_layers=3, n_model=self.ques_embed.n_out)

        self.encoder_n_hidden = self.ques_embed.n_out
        self.labeler = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True)
        self.crit = nn.CrossEntropyLoss()
    
    def forward(self, ques, doc, answerable=None, ans_label=None):
        x_q = self.ques_embed(ques)
        x_d = self.doc_embed(doc)
        q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)

        x_q = self.transformer_encoder(x_q, q_mask)
        x = self.transformer_decoder(x_d, x_q, d_mask, q_mask)

        pred_label = self.labeler(x)

        x_mean = x.mean(dim=1)
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