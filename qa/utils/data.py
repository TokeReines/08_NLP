import torch
import numpy as np
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
        ques, ans_start, ans, answerable, doc = list(self.data['question_text']), list(self.data['answer_start']), list(self.data['answer_text']), list(self.data['answerable']), list(self.data['document_plaintext'])
        
        ques = self.transform(ques)
        doc = self.transform(doc)
        for i in range(len(ans)):
            ans[i] = np.concatenate([[-1], ans[i], [-1]])

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
            a.append(torch.tensor(ans, dtype=torch.long))
            a_start.append(ans_start)
            ab.append(answerable)
        ques = pad(q, pad_index)
        doc = pad(docs, pad_index)
        ans_start = torch.tensor(a_start, dtype=torch.long)
        answerable = torch.tensor(ab, dtype=torch.long)
        ans = pad(a, -1)
        return ques, ans_start, ans, answerable, doc
    
class ConcatDataset(BertDataset):
    def fn_collate(self, data1):
        data, a, a_start, ab, mask = [], [], [], [], []
        pad_index = self.tokenizer.vocab[self.tokenizer.pad]
        for d in data1:
            ques, ans_start, ans, answerable, doc = d
            datum = ques + doc
            # q.append(pad([torch.tensor(ids, dtype=torch.long) for ids in ques], pad_index))
            # docs.append(pad([torch.tensor(ids, dtype=torch.long) for ids in doc], pad_index))
            data.append(pad([torch.tensor(ids, dtype=torch.long) for ids in datum], pad_index))
            m = [1 for _ in range(len(ques))] + [2 for _ in range(len(doc))]
            mask.append(torch.tensor(m, dtype=torch.long))
            a.append(torch.tensor(ans, dtype=torch.long))
            a_start.append(ans_start)
            ab.append(answerable)
        # ques = pad(q, pad_index)
        # doc = pad(docs, pad_index)
        data = pad(data, pad_index)
        mask = pad(mask, -1)
        ans_start = torch.tensor(a_start, dtype=torch.long)
        answerable = torch.tensor(ab, dtype=torch.long)
        ans = pad(a, -1)
        return data, ans_start, ans, answerable, mask
    