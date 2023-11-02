import torch
import torch.nn as nn

class ConvPooling(nn.Module):

    def __init__(self, c_in, c_out, kernel_size=7, stride=1, dropout=0.1):
        super(ConvPooling, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x, mask):
        x = x * torch.where(mask.unsqueeze(-1), 1., 0.)
        x = self.conv(x.transpose(1, 2))
        return self.dropout(x).transpose(2, 1)
    
class DeConvPooling(nn.Module):

    def __init__(self, c_in, c_out, kernel_size=7, stride=1, dropout=0.1):
        super(DeConvPooling, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride

        self.deconv = nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel_size, stride=stride)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        x = self.deconv(x)
        x = self.dropout(x).transpose(2, 1)
        x = x + torch.where(mask, 0., float('-inf'))
        return x