import torch
import torch.nn as nn
import copy
from modules.transformer import TransformerEncoderLayer, TransformerEncoder

class DualAttention(nn.Module):
    def __init__(self, layer, n_model, n_layers):
        super(DualAttention, self).__init__()
        self.n_layers = n_layers
        self.n_model = n_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    
    def forward(self, x, y, x_mask, y_mask, return_att=False):
        att_x, att_y = None, None
        for layer in self.layers:
            if return_att:
                x, y, att_x, att_y = layer(x, y, x_mask, y_mask, return_att=return_att)
            else:
                x, y = layer(x, y, x_mask, y_mask, return_att=return_att)
        if return_att:
            return x, y, att_x, att_y
        return x, y
    
class DualAttentionLayer(nn.Module):
    def __init__(self, n_model, att_dropout=0.2):
        super(DualAttentionLayer, self).__init__()
        self.n_model = n_model
        self.x_proj = nn.Linear(n_model * 2, n_model, bias=True)
        self.y_proj = nn.Linear(n_model * 2, n_model, bias=True)
        # self.y_proj = nn.Linear(n_model, n_model, bias=True)
        self.x_norm = nn.LayerNorm(n_model)
        self.y_norm = nn.LayerNorm(n_model)
        self.x_dropout = nn.Dropout(att_dropout)
        self.y_dropout = nn.Dropout(att_dropout)
        self.scale = n_model ** 0.5
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj.weight, 2 ** -0.5)
        nn.init.xavier_uniform_(self.y_proj.weight, 2 ** -0.5)
    
    def forward(self, x, y, x_mask, y_mask, return_att=False):
        l_t = torch.bmm(x, y.movedim(1, 2))
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, y_mask.shape[1])
        y_mask = y_mask.unsqueeze(1).repeat(1, x_mask.shape[1], 1)
        score = l_t
        att_x = torch.softmax(score + torch.where(y_mask, 0., float('-inf')), -1)
        att_x = self.x_dropout(att_x)
        att_y = torch.softmax(score + torch.where(x_mask, 0., float('-inf')), 1)
        att_y = self.y_dropout(att_y)

        new_y = torch.bmm(att_y.transpose(1, 2), x)
        new_x = torch.bmm(att_x, torch.cat([y, new_y], -1))

        new_x_ = torch.bmm(att_x, y)
        new_y_ = torch.bmm(att_y.transpose(1, 2), torch.cat([x, new_x_], -1))

        x = self.x_norm(x + self.x_proj(new_x))
        y = self.y_norm(y + self.y_proj(new_y_))
        # y = self.y_norm(y + self.y_proj(new_y))
        if return_att:
            return x, y, att_x, att_y
        return x, y
    
class TransformerDualAttention(nn.Module):
    def __init__(self, layer, n_model, n_layers):
        super(TransformerDualAttention, self).__init__()
        self.n_layers = n_layers
        self.n_model = n_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    
    def forward(self, x, y, x_mask, y_mask, return_att=False):
        att_x, att_y = None, None
        for layer in self.layers:
            if return_att:
                x, y, att_x, att_y = layer(x, y, x_mask, y_mask, return_att=return_att)
            else:
                x, y = layer(x, y, x_mask, y_mask, return_att=return_att)
        if return_att:
            return x, y, att_x, att_y
        return x, y
    
class TransformerDualAttentionLayer(nn.Module):
    def __init__(self, n_model, att_dropout=0.2):
        super(TransformerDualAttentionLayer, self).__init__()
        self.n_model = n_model
        self.x_proj = nn.Linear(n_model * 2, n_model, bias=True)
        self.y_proj = nn.Linear(n_model * 2, n_model, bias=True)
        self.x_norm = nn.LayerNorm(n_model)
        self.y_norm = nn.LayerNorm(n_model)
        self.x_dropout = nn.Dropout(att_dropout)
        self.y_dropout = nn.Dropout(att_dropout)
        layer = TransformerEncoderLayer(n_heads=8, n_model=n_model, n_inner=n_model)
        self.doc_self = TransformerEncoder(layer, n_layers=1, n_model=n_model)
        layer = TransformerEncoderLayer(n_heads=8, n_model=n_model, n_inner=n_model)
        self.ques_self = TransformerEncoder(layer, n_layers=1, n_model=n_model)
    
    def forward(self, x, y, x_mask, y_mask, return_att=False):
        l_t = torch.bmm(x, y.movedim(1, 2))
        x_mask_ = x_mask.unsqueeze(-1).repeat(1, 1, y_mask.shape[1])
        y_mask_ = y_mask.unsqueeze(1).repeat(1, x_mask.shape[1], 1)
        score = l_t
        att_x = torch.softmax(score + torch.where(y_mask_, 0., float('-inf')), -1)
        att_x = self.x_dropout(att_x)
        att_y = torch.softmax(score + torch.where(x_mask_, 0., float('-inf')), 1)
        att_y = self.y_dropout(att_y)
        new_y = torch.bmm(att_y.transpose(1, 2), x)
        new_x = torch.bmm(att_x, torch.cat([y, new_y], -1))

        new_x_ = torch.bmm(att_x, y)
        new_y_ = torch.bmm(att_y.transpose(1, 2), torch.cat([x, new_x_], -1))

        x = self.x_norm(x + self.x_proj(new_x))
        y = self.y_norm(y + self.y_proj(new_y_))

        x = self.ques_self(x, x_mask)
        y = self.doc_self(y, y_mask)
        if return_att:
            return x, y, att_x, att_y
        return x, y