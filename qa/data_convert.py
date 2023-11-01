from utils.fn import to_list_tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from bnlp import BasicTokenizer
import pandas as pd
import numpy as np

def convert_to_feather(language, path, tokenizer=None):

    dataset = load_dataset("copenlu/answerable_tydiqa")
    test_set = dataset["validation"].filter(lambda example, idx: example['language'] == language, with_indices=True)

    train_set = dataset["train"].filter(lambda example, idx: example['language'] == language, with_indices=True).to_pandas()
    train_set, dev_set = train_test_split(train_set, test_size=0.15, random_state=42, shuffle=True)
    test_set = test_set.to_pandas()

    l_name = ['question_text', 'answer_start', 'answer_text', 'answerable', 'document_plaintext']

    train_set = to_list_tuple(train_set, l_name, tokenizer)
    dev_set = to_list_tuple(dev_set, l_name, tokenizer)
    test_set = to_list_tuple(test_set, l_name, tokenizer)

    train_set.to_feather(f'./{path}/{language}_train.feather')
    dev_set.to_feather(f'./{path}/{language}_dev.feather')
    test_set.to_feather(f'./{path}/{language}_test.feather')

# # convert_to_feather('arabic', 'data')
# # convert_to_feather('indonesian', 'data')
# tokenizer = BasicTokenizer()
# convert_to_feather('bengali', 'data', tokenizer)

def get_start(ans):
    start = 0
    end = 0
    prev = 0
    for i in range(len(ans)):
        if prev == 0 and ans[i] == 1:
            start = i
        if prev == 1 and (ans[i] == 0 or ans[i] == -1):
            end = i-1
        prev = ans[i]
    # if end < start:
    #     print(start, end, ans)
    assert end >= start
    return start

def get_end(ans):
    start = 0
    end = 0
    prev = 0
    for i in range(len(ans)):
        if prev == 0 and ans[i] == 1:
            start = i
        if prev == 1 and (ans[i] == 0 or ans[i] == -1):
            end = i-1
        prev = ans[i]
    # if end < start:
    #     print(start, end, ans)
    assert end >= start
    return end

def add_start_end(language, path):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')

    train_set["answer_text"] = train_set["answer_text"].apply(lambda x: np.concatenate([[-1], x, [-1]]))
    dev_set["answer_text"] = dev_set["answer_text"].apply(lambda x: np.concatenate([[-1], x, [-1]]))
    test_set["answer_text"] = test_set["answer_text"].apply(lambda x: np.concatenate([[-1], x, [-1]]))

    train_set["start_pos"] = train_set.apply(lambda x: get_start(x["answer_text"]), 1)
    train_set["end_pos"] = train_set.apply(lambda x: get_end(x["answer_text"]), 1)

    dev_set["start_pos"] = dev_set.apply(lambda x: get_start(x["answer_text"]), 1)
    dev_set["end_pos"] = dev_set.apply(lambda x: get_end(x["answer_text"]), 1)

    test_set["start_pos"] = test_set.apply(lambda x: get_start(x["answer_text"]), 1)
    test_set["end_pos"] = test_set.apply(lambda x: get_end(x["answer_text"]), 1)

    train_set.to_feather(f'./{path}/{language}_train.feather')
    dev_set.to_feather(f'./{path}/{language}_dev.feather')
    test_set.to_feather(f'./{path}/{language}_test.feather')

def get_max(language):
    train_set = pd.read_feather(f'./data/{language}_train.feather')
    dev_set = pd.read_feather(f'./data/{language}_dev.feather')
    test_set = pd.read_feather(f'./data/{language}_test.feather')

    t = train_set.apply(lambda x: x['end_pos'] - x['start_pos'], 1)
    print(t.max())
    t = dev_set.apply(lambda x: x['end_pos'] - x['start_pos'], 1)
    print(t.max())
    t = test_set.apply(lambda x: x['end_pos'] - x['start_pos'], 1)
    print(t.max())

# add_start_end("indonesian", "data")
# add_start_end("arabic", "data")
# add_start_end("bengali", "data")
# get_max("indonesian")
# get_max("bengali")
# get_max("arabic")