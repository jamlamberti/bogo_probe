"""A simple tokenizer implementation using stopwords"""
from nltk.corpus import stopwords

def tokenize(msg):
    """Tokenize a string"""
    body = ''.join([x if x.isalnum() else ' ' for x in msg])
    tokens = [word.lower()
            for word in body.split()
            if word.lower() not in stopwords.words('english')]
    return tokens
