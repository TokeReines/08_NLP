from torch.utils.data import Dataset

class FFN2Dataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, question_mask = self.tokenizer(self.data[idx]['question_text'])
        document_ids, document_mask = self.tokenizer(self.data[idx]['document_plaintext'])
        label = self.data[idx]['annotations']['answer_start'][0] != -1
        return question_ids, question_mask, document_ids, document_mask, label