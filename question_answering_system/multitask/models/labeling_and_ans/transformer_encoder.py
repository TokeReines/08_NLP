import torch
import torch.nn as nn
from modules.bertembed import TransformerEmbedding
from modules.transformer import TransformerEncoder, TransformerEncoderLayer
from modules.mlp import MLP
from modules.labeling import Labeler
from models.basemodel import BaseModel

class TransformerEncoderLabelingAnswerable(BaseModel):

    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, n_layers_trans=1,**kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d, **kwargs)

        layer = TransformerEncoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer, n_layers=n_layers_trans, n_model=self.n_out)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        _, ques_len, _ = x_q.shape
        x = torch.cat([x_q, x_d], dim=1)
        x_mask = torch.cat([q_mask, d_mask], dim=1)
        x = self.transformer_encoder(x, x_mask)
        return x[:, :ques_len, :], x[:, ques_len:, :], None, None
