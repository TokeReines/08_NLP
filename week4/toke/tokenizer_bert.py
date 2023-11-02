from transformers import AutoTokenizer

class TokenizerBERT:
    def __init__(self, model_name="bert-base-uncased", max_seq_length=512, stride=50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_seq_length = max_seq_length
        self.stride = stride

    def __call__(self, question: str, document: str, return_offsets_mapping=True):
        encoded = self.tokenizer.encode_plus(
            question,
            document,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length",
            truncation="only_second",  # Only truncate the document, not the question
            return_offsets_mapping=return_offsets_mapping,
            stride=self.stride  # This is the sliding window stride
        )

        return encoded