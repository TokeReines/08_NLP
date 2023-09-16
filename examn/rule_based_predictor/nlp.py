

from collections import Counter

from scipy import spatial


def classifier_bow_cosine(question: str, document: str, threshold=0.4) -> bool:
    """Calculates the cosine distance between the Bag of Words vector for the question and the document. Returns True if the distance is less than some threshold"""
    
    def get_bow(words: list[str], word_to_ix: dict):
        """Returns the Bag of Words vector for the given text"""
        counter = {}
        counter = Counter(words)
        return [counter[word] for word in word_to_ix]
    
    question_words = question.split()
    document_words = document.split()
    word_set = set(question_words)
    word_to_ix = {word: i for i, word in enumerate(word_set)}
    
    question_vector = get_bow(question_words, word_to_ix)
    document_vector = get_bow(document_words, word_to_ix)
    similarity = 1 - spatial.distance.cosine(question_vector, document_vector)
    return similarity > threshold

classifier_bow_cosine("What is the capital of France?", "The capital of France is Paris. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Some more text here. And a little bit more.")
    