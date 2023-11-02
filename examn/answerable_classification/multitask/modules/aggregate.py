import torch
import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, x_q, x_d, new_q_mask, new_d_mask, batch_id, batch, ques_len, doc_len):
        new_x_q = x_q.new_zeros(batch, ques_len, self.n_out)
        new_x_d = x_d.new_zeros(batch, doc_len, self.n_out)
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
