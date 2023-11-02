import pandas as pd
from torch.utils.data import Dataset


class DatasetBERT(Dataset):
    def __init__(self, data, tokenizer, window_stride=128):
        self.data = data
        self.tokenizer = tokenizer
        self.window_stride = window_stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data['question_text'][index]
        document = self.data['document_plaintext'][index]
        answer_start = self.data['annotations'][index]['answer_start'][0]
        answer_text = self.data['annotations'][index]['answer_text'][0]
        answer_end = answer_start + len(answer_text)

        inputs = self.tokenizer(question, document, return_offsets_mapping=True)
        offsets = inputs["offset_mapping"][0].tolist()

        # Find start and end token position for answer
        token_start_index = -1
        token_end_index = -1

        for i, (start_char, end_char) in enumerate(offsets):
            if (start_char <= answer_start and answer_end <= end_char):
                if token_start_index == -1:
                    token_start_index = i
                token_end_index = i

        labels = {
            "answer_start": token_start_index,
            "answer_end": token_end_index
        }

        return inputs, labels
