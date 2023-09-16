class Transform():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: list[str]) -> list[list[str]]:
        res = []
        for seq in data:
            tokens = [self.tokenizer(text) for text in seq]
            res.append(tokens)
        return tokens