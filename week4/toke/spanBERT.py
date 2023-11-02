import numpy as np
import torch.nn as nn
import torch


class SpanBERT(nn.Module):
    def __init__(self, bert, tokenizer):
        super(SpanBERT, self).__init__()
        self.bert = bert
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None
    ):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            output_attentions=True
        )

        return outputs
