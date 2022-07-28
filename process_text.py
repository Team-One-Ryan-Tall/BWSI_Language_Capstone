from collections import Counter
import numpy as np
import re, string

def process_text(text):
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    strip = punc_regex.sub('', text)
    return sorted(Counter(strip.lower().split()))