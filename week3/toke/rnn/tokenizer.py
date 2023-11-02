from bpemb import BPEmb

class Tokenizer:
    def __init__(self, lang='en', vs=25000):
        self.bpemb = BPEmb(lang=lang, vs=vs)

    def __call__(self, text):
        return self.bpemb.encode_ids(text)