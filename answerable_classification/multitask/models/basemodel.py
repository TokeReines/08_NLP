import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding, TransformerEmbeddingQuesDoc
from modules.labeling import Labeler

class BaseModel(nn.Module):

    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, return_att=False):
        super(BaseModel, self).__init__()
        self.pad_index = pad_index
        self.finetune = finetune
        self.loss_weights = loss_weights
        self.embed_type = embed_type
        self.split_q_d = split_q_d
        self.return_att = return_att
        if embed_type == 'two-bert':
            self.ques_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
            self.doc_embed = TransformerEmbedding(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride)
            self.n_out = self.ques_embed.n_out
        else:
            self.embed = TransformerEmbeddingQuesDoc(name, n_layers=n_layers, pooling=pooling, pad_index=pad_index, mix_dropout=mix_dropout, finetune=finetune, stride=stride, dropout=encoder_dropout)
            self.doc_max_len = self.embed.doc_max_len
            self.doc_stride = self.embed.doc_stride
            self.n_out = self.embed.n_out
        
        self.encoder_n_hidden = self.n_out
        self.labeler = Labeler(self.encoder_n_hidden) 
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True) if not split_q_d else nn.Linear(self.encoder_n_hidden*2, 2, bias=True)
        nn.init.orthogonal_(self.classification.weight)
        nn.init.zeros_(self.classification.bias)
        self.crit = nn.CrossEntropyLoss()

    def forward(self, ques, doc, answerable, start, end):
        q_mask, d_mask = ques.ne(self.pad_index).any(-1), doc.ne(self.pad_index).any(-1)
        batch, ques_len = q_mask.shape
        _, doc_len = d_mask.shape
        m_len = q_mask.shape[1]
        x_q, x_d, batch_id, new_q_mask, new_d_mask = self.encode(ques, doc)
        if self.embed_type == 'two-bert':
            new_q_mask = q_mask
            new_d_mask = d_mask
        x_q, x_d, att_x, att_y = self.attention_score(x_q, x_d, new_q_mask, new_d_mask)
        # x_q, x_d = self.attention_score(x_q, x_d, new_q_mask, new_d_mask)
        x_q, x_d = self.aggregate(x_q, x_d, new_q_mask, new_d_mask, batch_id, batch, ques_len, doc_len)
        pred_answerable, pred_start, pred_end = self.scorer(x_q, q_mask, x_d, d_mask, m_len)
        if answerable is not None:
            loss = self.loss_compute(pred_answerable, pred_start, pred_end, answerable, start, end)
            if self.return_att:
                return loss, (pred_start, pred_end), pred_answerable, att_x, att_y
            else:
                return loss, (pred_start, pred_end), pred_answerable, None, None
        return (pred_start, pred_end), pred_answerable

    def encode(self, ques, doc):
        if self.embed_type == 'two-bert':
            x_q = self.ques_embed(ques)
            x_d = self.doc_embed(doc) 
            return x_q, x_d, None, None, None
        else:
            x_q, x_d, batch_id, new_q_mask, new_d_mask = self.embed(ques, doc)
            return x_q, x_d, batch_id, new_q_mask, new_d_mask
    
    def scorer(self, ques, q_mask, doc=None, d_mask=None, m_len=None):
        if doc is None:
            lens = torch.sum(q_mask, -1, keepdim=True)
            x_mean = torch.sum(ques * q_mask.unsqueeze(-1), dim=1) / lens
            pred_answerable = self.classification(x_mean)
            pred_start, pred_end = self.labeler(ques[:,m_len:,:], d_mask, pred_answerable)
        else:
            d_lens = torch.sum(d_mask, -1, keepdim=True)
            d_mean = torch.sum(doc * d_mask.unsqueeze(-1), dim=1) / d_lens
            q_lens = torch.sum(q_mask, -1, keepdim=True)
            q_mean = torch.sum(ques * q_mask.unsqueeze(-1), dim=1) / q_lens
            if self.split_q_d:
                pred_answerable = self.classification(torch.cat([q_mean, d_mean], -1))
            else:
                pred_answerable = self.classification(q_mean + d_mean)
            pred_start, pred_end = self.labeler(doc, d_mask, pred_answerable)
        return pred_answerable, pred_start, pred_end 

    def aggregate(self, x_q, x_d, new_q_mask, new_d_mask, batch_id, batch, ques_len, doc_len):
        if self.embed_type == 'two-bert':
            return x_q, x_d
        new_x_q = x_q.new_zeros(batch, ques_len, self.encoder_n_hidden)
        new_x_d = x_d.new_zeros(batch, doc_len, self.encoder_n_hidden)
        for i in range(batch):
            x_q_tmp = x_q[batch_id.eq(i)]
            x_q_ = torch.mean(x_q_tmp, 0)
            x_d_tmp = x_d[batch_id.eq(i)]
            x_d_ = x_d_tmp[0, :-1]
            d_mask_ = new_d_mask[batch_id.eq(i)]
            if x_d_tmp.shape[0] > 1:
                denom = x_d_tmp.new_ones(x_d_.shape[0])
                for j in range(1, x_d_tmp.shape[0]):
                    tmp = x_d_tmp[j][d_mask_[j]]
                    tmp = tmp[:-1]
                    x_d_[-(self.doc_max_len-self.doc_stride):] = x_d_[-(self.doc_max_len-self.doc_stride):] + tmp[:(self.doc_max_len-self.doc_stride)]
                    denom[-(self.doc_max_len-self.doc_stride):] += 1
                    x_d_ = torch.cat([x_d_, tmp[self.doc_max_len-self.doc_stride:]], 0)
                    denom = torch.cat([denom, x_d_.new_ones(tmp[self.doc_max_len-self.doc_stride:].shape[0])], 0)
                x_d_ = x_d_ / denom.unsqueeze(1)
            new_x_q[i] = x_q_
            new_x_d[i, :x_d_.shape[0], :] = x_d_
        
        return new_x_q, new_x_d
    
    def attention_score(self, x_q, x_d, q_mask, d_mask):
        raise NotImplementedError
    
    def loss_compute(self, pred_answerable, pred_start, pred_end, answerable, start, end):
        loss_answerable = self.crit(pred_answerable, answerable)
        loss_label = self.crit(pred_start, start) + self.crit(pred_end, end)
        loss = loss_answerable + loss_label if self.loss_weights is None else loss_answerable * self.loss_weights + (1 - self.loss_weights) * loss_label
        return loss

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