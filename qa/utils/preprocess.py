import re
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import unicodedata as ud
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def break_into_sentences(paragraph):
    sentences = list()
    temp_sentence = list()
    flag = False
    for ch in paragraph.strip():
        if ch in [u'؟', u'!', u'.', u':', u'؛']:
            flag = True
        elif flag:
            sentences.append(''.join(temp_sentence).strip())
            temp_sentence = []
            flag = False

        temp_sentence.append(ch)

    else:
        sentences.append(''.join(temp_sentence).strip())
        return sentences
    
def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, ' ', raw_html)
    return cleantext

def remove_ref(sentence):
    result = re.sub("(\[\s*\d+\s*\])", " ", sentence)
    return result

def clean_arabic(l_arabic):
    l_cleaned_arabic = []
    for p in l_arabic:
        ss = break_into_sentences(remove_ref(cleanhtml(p)))
        for s in ss:
            l_cleaned_arabic.append(s)
    return l_cleaned_arabic

def get_cleaned_sentence_list(l):
    cleaned = clean_arabic(l)
    res = []
    counts = []
    for i in cleaned:
        words = word_tokenize(i)
        counts.append(len(words))
        sent = ' '.join(words)
        res.append(sent)
    return res, counts

def get_tokens(docs):
    res = []
    for i in docs:
        words = word_tokenize(i)
        res.append(words)
    
    return res

def tokenize_cell(doc):
    return wordpunct_tokenize(doc)

# def split_punct(doc):
#     res = ""
#     for i, c in enumerate(doc):
#         if ud.category(c).startswith('P'):
#             if i == 0:
#                 res += c
#             elif i == len(doc) - 1:
#                 res += ' '
#                 res += c
            
#         else:
#            res += c 