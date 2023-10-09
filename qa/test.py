from datasets import load_dataset
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import pandas as pd
from transformers import AutoTokenizer
from bnlp import BasicTokenizer
train_set = pd.read_feather(f'./data/bengali_train.feather')

# print(word_tokenize('1905.'))

c = 0

dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"].filter(lambda example, idx: example['language'] == 'bengali', with_indices=True).to_pandas()

sample = train_set.iloc[0]["document_plaintext"]
t = BasicTokenizer()(sample)



tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base", local_files_only=False)

print(t)
print(tokenizer.tokenize(sample))
print([tokenizer.tokenize(i) for i in t])
# for i in range(len(train_set.index)):
#     d = train_set["document_plaintext"].iloc[i]
#     # a = train_set["annotations"].iloc[i]["answer_text"][0]
#     # s = train_set["annotations"].iloc[i]["answer_start"][0]

#     # d = ' '.join(wordpunct_tokenize(d))
#     # a = ' '.join(wordpunct_tokenize(a))
    
#     print(d)
#     a = [tokenizer.tokenize(i) for i in d]
#     print(a)
#     break
#         # print('error')
#         # print("doc", d[s:s+len(a)])
#         # print("ans", a)

# # print(c)