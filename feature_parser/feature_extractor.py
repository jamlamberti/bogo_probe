import os
import email
import numpy as np
from collections import Counter
from .tokenizer import tokenize

def load_email(email_file):
    """Load an email"""
    with open(email_file, 'r') as f_handler:
        msg = email.message_from_file(f_handler)
        return msg.get_payload()


def parse_dataset(direc, tokenizer):
    tokens = []
    for root, _, files in os.walk(direc):
        tokens.extend([tokenizer(load_email(os.path.join(root, x)))
            for x in files])
    return tokens


def tf_idf(doc, world):
    d_cnt = Counter(doc)
    scores = []
    for tok in d_cnt.keys():
        tf = d_cnt[tok]*1./len(doc)
        idf = np.log(len(world) * 1.0/sum([1 for w in world if tok in w]))
        scores.append([tok, tf*idf])

    return scores


def generate_feat_vec(doc, features):
    d_cnt = Counter(doc)
    return [d_cnt[f] for f in features]


def feature_extractor(ham_toks, spam_toks, max_features):
    ham_joined = sum(ham_toks, [])
    spam_joined = sum(spam_toks, [])
    h_feats = sorted(
            tf_idf(ham_joined, ham_joined + spam_joined),
            key=lambda x: x[1],
            reverse=True)[:max_features/2]

    s_feats = sorted(
            tf_idf(spam_joined, ham_joined + spam_joined),
            key=lambda x: x[1],
            reverse=True)[:max_features/2]

    return h_feats, s_feats

def feature_parser(ham_dir, spam_dir, max_features):
    ham_toks = parse_dataset(ham_dir, tokenize)
    spam_toks = parse_dataset(spam_dir, tokenize)
    h_feats, s_feats = feature_extractor(ham_toks, spam_toks, max_features)
    features = [i[0] for i in h_feats] + [i[0] for i in s_feats]
    print h_feats
    h_vec = [generate_feat_vec(doc, features) for doc in ham_toks]
    s_vec = [generate_feat_vec(doc, features) for doc in spam_toks]

    return h_vec, s_vec
