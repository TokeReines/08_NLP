import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TokenizerTFIDF():
    def __init__(self, data, max_features=1000, separator="<SEP>"):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Create vocab from data with max_length length
        self.create_vocab(data, separator)
        
    def create_vocab(self, data, separator):
        # Concat data['question_text'] and data['document_plaintext']
        text = [" ".join(question) + separator + " ".join(document) for question, document in zip(data['question_text'], data['document_plaintext'])]
        self.vectorizer.fit(text)
        
    def __call__(self, text) -> list[int]:
        tokens = self.vectorizer.transform([text]).toarray()
        return tokens[0]