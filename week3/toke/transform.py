class Transform():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # def __call__(self, data: list[str]) -> list[list[str]]:
    #     res = []
    #     for seq in data:
    #         tokens = [self.tokenizer(text) for text in seq]
    #         res.append(tokens)
    #     return res
    
    def __call__(self, data: list[str]) -> list[list[str]]:
        max_length = max(len(seq) for seq in data)  # Find the maximum length of sequences in the data
        
        res = []
        for seq in data:
            tokens = [self.tokenizer(text) for text in seq]
            padded_tokens = [[tokens[i]] + [0] * (max_length - len(tokens)) for i in range(len(tokens))]  # Pad sequences with zeros to make them equal length
            res.append(padded_tokens)
        return res
    