"""A feature extractor implementation"""

import os
import email
from collections import Counter
import numpy as np
from .tokenizer import tokenize


def load_email(email_file):
    """Load an email"""
    with open(email_file, 'r') as f_handler:
        msg = email.message_from_file(f_handler)
        return msg.get_payload()


def parse_dataset(direc, tokenizer):
    """Parse a dataset, and apply a certain tokenizer"""
    tokens = []
    for root, _, files in os.walk(direc):
        tokens.extend([tokenizer(load_email(os.path.join(root, x)))
                       for x in files])
    return tokens


def tf_idf(doc, world):
    """
    Run tf-idf for all tokens in the document
    against the world
    """
    d_cnt = Counter(doc)
    scores = []
    for tok in d_cnt.keys():
        tf_score = d_cnt[tok] * 1. / len(doc)
        idf = np.log(len(world) * 1.0 / sum([1 for w in world if tok in w]))
        scores.append([tok, tf_score * idf])
    return scores


def word_counter(ham_dir, spam_dir):
    """
    Count the tokens similiar to SB...
    This works well for a Bayesian Classifier, but not so hot for others
    """
    ham_toks = parse_dataset(ham_dir, tokenize)
    spam_toks = parse_dataset(spam_dir, tokenize)
    global_tokens = set().union(*ham_toks).union(*spam_toks)

    # Compute the overall counts
    tokens = {tok: [
        sum([x.count(tok) for x in ham_toks]),
        sum([x.count(tok) for x in spam_toks])] for tok in global_tokens}

    return tokens


def generate_feat_vec(doc, features):
    """Generate the feature vector for a given document"""
    d_cnt = Counter(doc)
    return [d_cnt[f] for f in features]


def feature_extractor(ham_toks, spam_toks, max_features):
    """Extract max_features features from ham and spam"""
    ham_joined = sum(ham_toks, [])
    spam_joined = sum(spam_toks, [])

    # TODO: Should max_features be split evenly or should we do it against
    # a global order

    world = [ham_joined, spam_joined]

    h_feats = sorted(
        tf_idf(ham_joined, world),
        key=lambda x: x[1],
        reverse=True)[:max_features / 2]

    s_feats = sorted(
        tf_idf(spam_joined, world),
        key=lambda x: x[1],
        reverse=True)[:max_features / 2]

    return h_feats, s_feats


def feature_parser(ham_dir, spam_dir, max_features):
    """
    Returns a feature vector for ham and spam emails of dim max_features
    """
    ham_toks = parse_dataset(ham_dir, tokenize)
    spam_toks = parse_dataset(spam_dir, tokenize)

    h_feats, s_feats = feature_extractor(ham_toks, spam_toks, max_features)
    features = [i[0] for i in h_feats] + [i[0] for i in s_feats]

    h_vec = [generate_feat_vec(doc, features) for doc in ham_toks]
    s_vec = [generate_feat_vec(doc, features) for doc in spam_toks]

    return h_vec, s_vec

if __name__ == '__main__':
    word_counter('data/test/ham/', 'data/test/spam')
