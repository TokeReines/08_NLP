import torch
from utils.preprocess import tokenize_cell, remove_ref, cleanhtml
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

def pad(
    tensors,
    padding_value = 0,
    total_length = None,
    padding_side = 'right'
):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor

def to_list_tuple(df, l_name, tokenizer):
    f = tokenizer
    if tokenizer is None:
        f = tokenize_cell
    df['answerable'] = df.apply(lambda x: 1 if x['annotations']['answer_start'][0] != -1 else 0, axis=1)
    df['answer_start'] = df.apply(lambda x: x['annotations']['answer_start'][0], axis=1)
    df['answer_text'] = df.apply(lambda x: x['annotations']['answer_text'][0], axis=1)
    df['answer_text'] = df['answer_text'].apply(lambda x: f(x))
    df['document_plaintext'] = df['document_plaintext'].apply(lambda x: f(x))
    df['question_text'] = df['question_text'].apply(lambda x: f(x))
    df['answer_text'] = df.apply(lambda x: label_seq(x['answer_text'], x['document_plaintext'], x['answerable']), axis=1)
    # for i in range(len(df.index)):
    #     df['answer_text'].iloc[i] = label_seq(df['answer_text'].iloc[i], df['document_plaintext'].iloc[i], df['answerable'].iloc[i])
    return df[l_name]
    # return list(df[l_name].itertuples(index=False))
    # pass

def label_seq(answer_text, document_plaintext, answerable):
    res = [0 for _ in range(len(document_plaintext))]
    if answerable == 0:
        return res
    index = -1
    for i in range(len(document_plaintext)):
        d = document_plaintext[i]
        if answer_text[0] in d and document_plaintext[i+1:i+len(answer_text)-1] == answer_text[1:len(answer_text)-1] and (answer_text[-1] in document_plaintext[i+len(answer_text)-1]):
        # if answer_text[0] == document_plaintext[i] and answer_text == document_plaintext[i:i+len(answer_text)]:
            index = i
            break
    if index == -1:
        print("error in pattern matching")
        print('answer', answer_text)
        print(document_plaintext)
    else:
        res[index:index+len(answer_text)] = [1 for _ in range(len(answer_text))]
    # print(res)
    return res
