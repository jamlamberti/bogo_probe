from feature_parser.feature_extractor import load_email
from auto_evader.markov_model import Markov
from auto_evader.train_model import get_sentences
import os

def getAllText(direc):
    """Iterate through directories and remove all noise"""
    l = []
    for root, _, files in os.walk(direc):
        l.extend([''.join(load_email(os.path.join(root,x)).splitlines())
                  for x in files])

    return ''.join(l)

def main(direc):

    sent  = getAllText(direc)
    markc = Markov()

    for sentence in get_sentences(sent):
        markc.learn(sentence.raw)

    print markc.ask()
    print markc.ask()

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
