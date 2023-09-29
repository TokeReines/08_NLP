import torch
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from utils.preprocess import tokenize_cell

def pad(
    tensors: List[torch.Tensor],
    padding_value: int = 0,
    total_length: int = None,
    padding_side: str = 'right'
) -> torch.Tensor:
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor

def to_list_tuple(df, l_name):
    df['answerable'] = df.apply(lambda x: 1 if x['annotations']['answer_start'][0] != -1 else 0, axis=1)
    df['answer_start'] = df.apply(lambda x: x['annotations']['answer_start'][0], axis=1)
    df['answer_text'] = df.apply(lambda x: x['annotations']['answer_text'][0], axis=1)
    df['document_plaintext'] = df['document_plaintext'].apply(lambda x: tokenize_cell(x))
    df['question_text'] = df['question_text'].apply(lambda x: tokenize_cell(x))
    return df[l_name]
    # return list(df[l_name].itertuples(index=False))
    # pass
        