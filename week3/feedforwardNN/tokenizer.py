import nltk
from nltk.tokenize import word_tokenize

class Tokenizer:
    def __init__(self):
        nltk.download('punkt')
        self.tokenizer = word_tokenize        
        self.vocab = set()

    def __call__(self, text: str):
        tokens = self.tokenizer(text)
        self.vocab.update(tokens)
        return tokens
    
    def vocab(self):
        return list(self.vocab)