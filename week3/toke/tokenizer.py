from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize

# class Tokenizer:
#     def __init__(self, separator="<SEP>"):
#         nltk.download('punkt')
#         self.separator = separator
#         self.tokenizer = word_tokenize        
#         self.vocab = Vocab(specials=[separator])

#     def __call__(self, text: str):
#         tokens = self.tokenizer(text)
#         self.vocab.update(tokens)
#         return tokens
    
#     def vocab(self):
#         return list(self.vocab)
    
    
class Tokenizer:
    def __init__(self, data, separator="<SEP>"):
        self.counter = Counter()
        self.vocab = self.build_vocab(data, separator)
        self.separator = separator

    def build_vocab(self, data, separator):
        counter = Counter()
        for _, row in data.iterrows():
            counter.update(row['question_text'])
            counter.update(row['document_plaintext'])
            
        # v = vocab(counter, specials=['<UNK>', separator])
        # v.set_default_index(v['<UNK>'])
        # return v

    def __call__(self, text: str) -> list[int]:
        tokens = text.split()  # Split text into individual tokens
        indices = [self.vocab[token] for token in tokens]  # Convert tokens to indices using the vocabulary
        return indices[0]