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