"""wrapper around markov_model"""
from __future__ import print_function
from textblob import TextBlob
from .markov_model import Markov


def get_sentences(sample):
    """Helper function to extract sentences"""
    data = ''
    with open(sample, 'r') as f_handle:
        data = f_handle.read().decode('ascii', 'ignore')

    blob = TextBlob(data.replace('\n', ' ').replace(
        '"', '').replace('--', ' '))  # .replace('\'', ''))
    return blob.sentences


def main(dest, training_data):
    """driver for markov chain"""
    markc = Markov()
    for sample in training_data:
        for sentence in get_sentences(sample):
            markc.learn(sentence.raw)
    print(markc.ask())
    print(markc.ask())
    markc.dump_model(dest)
