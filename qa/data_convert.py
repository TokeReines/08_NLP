from utils.fn import to_list_tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from bnlp import BasicTokenizer

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

# convert_to_feather('arabic', 'data')
# convert_to_feather('indonesian', 'data')
tokenizer = BasicTokenizer()
convert_to_feather('bengali', 'data', tokenizer)