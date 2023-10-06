from datasets import load_dataset
from nltk.tokenize import word_tokenize, wordpunct_tokenize

dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"].filter(lambda example, idx: example['language'] == 'arabic', with_indices=True).to_pandas()

print(word_tokenize('1905.'))

c = 0

for i in range(len(train_set.index)):
    d = train_set["document_plaintext"].iloc[i]
    a = train_set["annotations"].iloc[i]["answer_text"][0]
    s = train_set["annotations"].iloc[i]["answer_start"][0]

    d = ' '.join(wordpunct_tokenize(d))
    a = ' '.join(wordpunct_tokenize(a))
    

    if not a in d:
        c += 1
        # print('error')
        # print("doc", d[s:s+len(a)])
        # print("ans", a)

print(c)