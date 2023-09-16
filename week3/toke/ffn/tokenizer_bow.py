# Define the BOWTokenizer class
from collections import Counter


class TokenizerBOW():
    def __init__(self, data, max_length, separator="<SEP>"):
        self.separator = separator
        self.vocab_list = [separator, '<UNK>']
        self.vocab_dict = {}
        
        # Create vocab from data with max_length length
        self.create_vocab(data, max_length)
        
    def create_vocab(self, data, max_length):
        counter = Counter()
        # Create vocab from counter
        for question in data['question_text']:
            tokens = self._make_tokens(" ".join(question))
            counter.update(tokens)

        for document in data['document_plaintext']:
            tokens = self._make_tokens(" ".join(document))
            counter.update(tokens)

        self.vocab_list += [k for k, v in counter.most_common(max_length - 2)]
        self.vocab_dict = {k: i for i, k in enumerate(self.vocab_list)}
        
    def _make_tokens(self, text: str):
        return text.split()
    
    def __call__(self, text) -> list[int]:
        tokens = self._make_tokens(text)
        return [self.vocab_dict.get(token, self.vocab_dict['<UNK>']) for token in tokens]