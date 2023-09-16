# Define the Transform class
import numpy as np


class Transform():
    def __init__(self, tokenizer, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, data: list[list[str]]) -> list[int]:
        tokens = self.tokenizer(" ".join(data))
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = np.pad(tokens, (0, self.max_length - len(tokens)), mode='constant')
        return tokens