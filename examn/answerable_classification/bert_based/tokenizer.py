
from transformers import AutoTokenizer


class FFN2Tokenizer:
    def __init__(self, pretrained_model, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, max_length=max_length)
        self.max_length = max_length

    def __call__(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        return inputs["input_ids"], inputs["attention_mask"]
