import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import DualAttention, DualAttentionLayer
from modules.transformer import TransformerEncoder, TransformerEncoderLayer
from modules.mlp import MLP
from modules.convpooling import ConvPooling, DeConvPooling
from modules.labeling import Labeler
from models.basemodel import BaseModel

class ConvTransformerEncoderLabelingAnswerable(BaseModel):
    def __init__(self, embed_type, name, n_layers=4, pooling='mean', pad_index=0, mix_dropout=0., encoder_dropout=0., dropout=0., stride=128, finetune=False, loss_weights=None, split_q_d=True, conv_stride=64, n_layers_trans=1, **kwargs):
        super().__init__(embed_type, name, n_layers, pooling, pad_index, mix_dropout, encoder_dropout, dropout, stride, finetune, loss_weights, split_q_d, **kwargs)

        self.conv1 = ConvPooling(self.n_out, 1024, kernel_size=conv_stride, stride=conv_stride, dropout=0.1)

        layer = TransformerEncoderLayer(n_heads=8, n_model=self.n_out, n_inner=1024)
        self.transformer_encoder = TransformerEncoder(layer, n_layers=n_layers_trans, n_model=self.n_out)

        layer = TransformerEncoderLayer(n_heads=8, n_model=self.conv1.c_out, n_inner=1024)
        self.conv_transformer_encoder = TransformerEncoder(layer, n_layers=n_layers_trans, n_model=self.conv1.c_out)

        self.encoder_n_hidden = self.n_out + self.conv1.c_out
        self.labeler = Labeler(self.encoder_n_hidden) 
        self.classification = nn.Linear(self.encoder_n_hidden, 2, bias=True) if not split_q_d else nn.Linear(self.encoder_n_hidden*2, 2, bias=True)
        nn.init.orthogonal_(self.classification.weight)
        nn.init.zeros_(self.classification.bias)

    def attention_score(self, x_q, x_d, q_mask, d_mask):
        batch, ques_len, _ = x_q.shape
        x = torch.cat([x_q, x_d], dim=1)
        x_mask = torch.cat([q_mask, d_mask], dim=1)
        _, org_len = x_mask.shape

        pad_size = self.conv1.stride - (org_len % self.conv1.stride) if org_len % self.conv1.stride != 0 else 0
        x_conv = F.pad(x.transpose(1, 2), pad=(0, pad_size)).transpose(2, 1)
        x_conv_mask = F.pad(x_mask, pad=(0, pad_size), value=0)
        x_conv = self.conv1(x_conv, x_conv_mask)
        b1, b2, _ = x_conv.shape
        x_conv_mask = x_conv.new_ones((b1, b2), dtype=torch.bool)
        x_conv = self.conv_transformer_encoder(x_conv, x_conv_mask)
        x_conv = x_conv.tile(1, 1, self.conv1.stride).reshape(batch, -1, self.conv1.c_out)[:, :org_len, :] * torch.where(x_mask.unsqueeze(-1), 1., 0.)
        
        x = self.transformer_encoder(x, x_mask)
        x = torch.cat([x, x_conv], -1)
        return x[:, :ques_len, :], x[:, ques_len:, :], None, None
    