from collections import Counter
import numpy as np
import re, string

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def tokenize(in_string):
    strip = punc_regex.sub('', in_string)
    return strip.lower().split()

def create_bag_of_words(text, vocabulary=10000) -> Counter:
    return Counter(tokenize(text)).most_common(vocabulary)

def InverseFrequency(frequencies: Counter, doc_count: int):
    N = doc_count
    # nt = np.array(nt, dtype=float)
    return {words : np.log10(N/counts) for words, counts in frequencies.most_common()}