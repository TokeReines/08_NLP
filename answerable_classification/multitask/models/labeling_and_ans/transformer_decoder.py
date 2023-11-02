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