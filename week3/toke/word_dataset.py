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
        self.word_to_idx = {word: idx for idx, word in enumerate(tokenizer.vocab | set(":"))}
        self.chunk()
        
    def chunk(self):
        """
        Transform data and chunk it into a list of tuples
        """
        ques, ans_start, ans, answerable, doc = list(self.data['question_text']), list(self.data['answer_start']), list(self.data['answer_text']), list(self.data['answerable']), list(self.data['document_plaintext'])
        ques = self.transform(ques)
        doc = self.transform(doc)
        separator = np.array([self.word_to_idx[':']], dtype=np.float32)

        self.data = list(zip(ques, ans_start, ans, answerable, doc, separator))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, int, str, bool, np.ndarray]:
        question, ans_start, ans, answerable, document, separator = self.data[idx]
        
        return question, ans_start, ans, answerable, document, separator

    def text_to_bow(self, tokens) -> np.ndarray:
        # Tokenize the text and convert it to a Bag of Words representation
        bow = np.zeros(len(self.tokenizer.vocab), dtype=np.float32)
        for token in tokens:
            if token in self.word_to_idx:
                bow[self.word_to_idx[token]] += 1
        return bow