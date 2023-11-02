import torch
from transformers import BertTokenizer

class FFN2Tokenizer:
    def __init__(self, pretrained_model='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length
    
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
        return inputs['input_ids'], inputs['attention_mask']