# Define the Transform class
class TransformTFIDF():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: list[list[str]]) -> list[int]:
        tokens = self.tokenizer(" ".join(data))      
        return tokens