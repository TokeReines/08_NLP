from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Union, Iterable

class TransformerTokenizer:

    def __init__(self, name) -> TransformerTokenizer:
        from transformers import AutoTokenizer
        self.name = name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str) -> List[str]:
        from tokenizers.pre_tokenizers import ByteLevel
        if isinstance(self.tokenizer.backend_tokenizer.pre_tokenizer, ByteLevel):
            text = ' ' + text
        return tuple(i.strip() for i in self.tokenizer.tokenize(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __getstate__(self) -> Dict:
        return self.__dict__

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)

    @property
    def vocab(self):
        return defaultdict(lambda: self.tokenizer.vocab[self.unk],
                           {**self.tokenizer.get_vocab(), **self.tokenizer.get_added_vocab()})

    @property
    def tokens(self):
        return sorted(self.vocab, key=lambda x: self.vocab[x])

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad(self):
        return self.tokenizer.pad_token

    @property
    def unk(self):
        return self.tokenizer.unk_token

    @property
    def bos(self):
        return self.tokenizer.cls_token

    @property
    def eos(self):
        return self.tokenizer.sep_token

    def decode(self, text: List) -> str:
        return self.tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def extend(self, data: Iterable[str], length: int = 32000) -> TransformerTokenizer:
        t = self.tokenizer.train_new_from_iterator(data, length)
        self.tokenizer.add_tokens(list(set(t.get_vocab()) - set(self.vocab)))
        return self