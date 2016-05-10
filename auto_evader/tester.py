from feature_parser.feature_extractor import load_email
from auto_evader.markov_model import Markov
from auto_evader.train_model import get_sentences
import os
import email
from spambayes.tokenizer import Tokenizer
from spambayes.tokenizer import tokenize
from spambayes import TestDriver
from spambayes import msgs

def getAllText(direc):
    """Iterate through directories and remove all noise"""
    fin = []
    for root, _, files in os.walk(direc):
        for x in files:
            fp = open(os.path.join(root,x))
            msg = email.message_from_file(fp)
            t = Tokenizer()
            body = t.tokenize_body(msg)
            body = set(body)
            body.add('.')
            body.add('!')
            body.add('?')
            fin.append(' '.join(filter(lambda y: y in body, msg.__str__().split())))

    return fin

def trainTest(dirs):
    SpamDirs = ['/home/ems316/spambayes-1.1a6/utilities/Data/Spam/Set1/', '/home/ems316/spambayes-1.1a6/utilities/Data/Spam/Set2/']
    HamDirs = ['/home/ems316/spambayes-1.1a6/utilities/Data/Ham/Set1/', '/home/ems316/spambayes-1.1a6/utilities/Data/Ham/Set2/']

    ss = msgs.SpamStream("%s-%d" % (SpamDirs[1], 2), SpamDirs[1:], train=1)
    hs = msgs.HamStream("%s-%d" % (HamDirs[1], 2), HamDirs[1:], train=1)

    d = TestDriver.Driver()
    d.train(hs, ss)
    
    return d

def main(direc):

    sent  = getAllText(direc)
    markc = Markov()
    nBayes = trainTest('/home')
    for x in sent:
        markc.learn(x)
        
    if markc.memory[markc.get_initial()].index('.') > 0:
        markc.memory[markc.get_initial()].remove('.')
    print markc.ask()
    #print nBayes.classifier.chi2_spamprob(Probe(markc.ask().split()))
#    markc.dump_model()

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
