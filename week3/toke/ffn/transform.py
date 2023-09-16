# Define the Transform class
class Transform():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: list[str]) -> list[int]:
        tokens = [self.tokenizer(" ".join(d)) for d in data]        
        return tokens