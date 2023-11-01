import torch
import torch.nn as nn

class Labeler(nn.Module):
    def __init__(self, n_in):
        super(Labeler, self).__init__()
        self.l_start = nn.Linear(n_in, 1)
        self.l_end = nn.Linear(n_in + 1, 1)
    
    def forward(self, x, mask, answerable):
        # mask = mask.unsqueeze(-1)
        # _, seq_len, _ = x.shape
        # answerable = torch.softmax(answerable, -1).unsqueeze(1).repeat(1, seq_len, 1)
        # x = torch.cat([x, answerable], -1)
        start = self.l_start(x)
        start = start.squeeze(-1) + torch.where(mask, 0.0, float('-inf'))
        end = self.l_end(torch.cat([x, torch.softmax(start, -1).unsqueeze(-1)], -1))
        return start, end.squeeze(-1) + torch.where(mask, 0.0, float('-inf'))
    
    
