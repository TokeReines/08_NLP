from torch.utils.data import Dataset
import numpy as np

class BOWDataset(Dataset):
    def __init__(self, tokenizer, transform, data):
        """
        transform: 
        """
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = data
        self.chunk()
        
    def chunk(self):
        """
        Transform data and chunk it into a list of tuples
        """
        ques, ans_start, ans, answerable, doc = list(self.data['question_text']), list(self.data['answer_start']), list(self.data['answer_text']), list(self.data['answerable']), list(self.data['document_plaintext'])
        ques = self.transform(ques)
        doc = self.transform(doc)

        self.data = list(zip(ques, ans_start, ans, answerable, doc))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, int, str, bool, np.ndarray]:
        return self.data[idx]