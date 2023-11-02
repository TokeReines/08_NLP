import torch
from torch.utils.data import Dataset
import numpy as np

    
class Dataset(Dataset):
    def __init__(self, data, transform, tokenizer):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.chunk()
        
    def chunk(self):
        """
        Transform data and chunk it into a list of tuples
        """
        ques, answerable, doc = list(self.data['question_text']), list(self.data['answerable']), list(self.data['document_plaintext'])
        
        # Should just be tokens here (int list)
        
        self.data = list(zip(ques, answerable, doc))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, int, str, bool, np.ndarray]:
        question, answerable, document = self.data[idx]     
        input = self.transform(np.concatenate((question, ['<SEP>'], document)))
        return torch.tensor(input), torch.tensor(answerable)
    