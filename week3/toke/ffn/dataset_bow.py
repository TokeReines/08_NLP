import torch
from torch.utils.data import Dataset
import numpy as np

    
class DatasetBOW(Dataset):
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
        combined = ques + doc
        self.vocabulary = self.transform(combined)
        
        # Should just be tokens here (int list)
        ques = self.transform(ques)
        doc = self.transform(doc)
        
        # Vocabulary is built after transforming the data
        self.data = list(zip(ques, answerable, doc))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, int, str, bool, np.ndarray]:
        question, answerable, document = self.data[idx]

        # Convert text to Bag of Words embeddings
        question_bow = self.text_to_bow(question)
        document_bow = self.text_to_bow(document)
        
        return torch.tensor(question_bow), torch.tensor(answerable), torch.tensor(document_bow)
    
    def text_to_bow(self, tokens: list[int]) -> np.ndarray:
        # Tokenize the text and convert it to a Bag of Words representation
        bow = np.zeros(len(self.tokenizer.vocab_list), dtype=np.float32)
        for token in tokens:
            bow[token] = 1
            
        return bow