import torch
from utils.fn import pad
from tqdm import tqdm

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, transform, max_len, data, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = data
        self.max_len = max_len
        self.chunk()

    def chunk(self):
        # for d in tqdm(self.data):
        # for i in tqdm(range(len(self.data.index))):
        #     d = self.data.iloc[i]
        #     ques, ans_start, ans, answerable, doc = d['question_text'], d['answer_start'], d['answer_text'], d['answerable'], d['document_plaintext']
        #     ques = self.transform(ques)
        #     doc = self.transform(doc)
        #     print(doc.shape)
        #     res.append((ques, ans_start, ans, answerable, doc))
        ques, ans_start, ans, answerable, doc = list(self.data['question_text']), list(self.data['answer_start']), list(self.data['answer_text']), list(self.data['answerable']), list(self.data['document_plaintext'])
        ques = self.transform(ques)
        doc = self.transform(doc)

        self.data = list(zip(ques, ans_start, ans, answerable, doc))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def fn_collate(self, data):
        q, a, a_start, ab, docs = [], [], [], [], []
        pad_index = self.tokenizer.vocab[self.tokenizer.pad]
        for d in data:
            ques, ans_start, ans, answerable, doc = d
            q.append(pad([torch.tensor(ids, dtype=torch.long) for ids in ques], pad_index))
            docs.append(pad([torch.tensor(ids, dtype=torch.long) for ids in doc], pad_index))
            a.append(ans)
            a_start.append(ans_start)
            ab.append(answerable)
        ques = pad(q, pad_index)
        doc = pad(docs, pad_index)
        ans_start = torch.tensor(a_start, dtype=torch.long)
        answerable = torch.tensor(ab, dtype=torch.long)
        ans = a 
        return ques, ans_start, ans, answerable, doc
    
