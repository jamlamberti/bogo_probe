"""A simple Markov Chain implementation"""
import random
import json
from collections import defaultdict
from textblob import TextBlob


class Markov(object):

    """Markov class for building simple models over sentences"""
    memory = defaultdict(list)
    separator = ' '
    end_of_sentence = ['?', '.', '!']

    def __init__(self, multi=5):
        self.multi_hop_param = multi


    def learn(self, txt):
        """Train the model over a sentence"""
        for key, value in self.break_text(txt):
            self.memory[key].append(value)



    def ask(self, seed=None):
        """Generate a sentence"""
        ret = []
        if not seed:
            seed = self.get_initial()
        #cnt = 0;
        while True:
            link = self.step(seed)
            if link is None:
                ##ret[-1] = ret[-1] + '.'
                #if cnt > 100:
                break
                #seed = self.get_initial()
            elif link in self.end_of_sentence:
                ret[-1] = ret[-1] + link
                #if cnt > 100:
                 #   break
                #seed = self.get_initial()
            else:
                ret.append(link[0])
                seed = link[1]
            #cnt += 1
        return self.separator.join(ret)



    def tokenize(self, txt):
        """Tokenize a sentence"""
        blob = TextBlob(txt)
        prev = self.get_initial()[0]
        if len(blob.words):
            for word in blob.words:
                yield prev, word
                prev = word
            eos = '.'
            for char in self.end_of_sentence:
                temp = txt.rfind(char)
                if temp > 0:
                    eos = char
                    break
            yield prev, eos

    def break_text(self, txt):
        """Break text into tokens"""
        prev = list(self.get_initial())
        for new_word in txt.strip('"').strip().split(self.separator):
            word = new_word.strip('"').strip()
            yield tuple(prev), word
            prev = prev[1:] + [word]
        # yield tuple(prev[1:] + [word])

    def step(self, state):
        """Have the model take a single step and return the new state"""
        choice = random.choice(self.memory[state] or [''])
        if not choice:
            return None
        next_state = tuple(list(state[1:]) + [choice])
        return choice, next_state

    def get_initial(self):
        """Returns the inital state vector"""
        return tuple(['' for _ in range(self.multi_hop_param)])

    def dump_model(self, outfile='models/model.json'):
        """Dump the model into a json format"""

        def smart_join(k):
            """Join tuples"""
            if isinstance(k, tuple):
                return self.separator.join(k)
            return k

        def sort_alphabetically(lis):
            """Sort the list"""
            return list(set(sorted(lis)))

        def sort_by_cnt(lis):
            """Sort by # of occurrences, then alphabetically"""
            # return filter(
            #    lambda s: len(s) > 0,
            #    set(sorted(
            #        lis,
            #        key=lambda x: (lis.count(x), x),
            #        reverse=True)))
            sorted_list = sorted(
                list(set(lis)),
                key=lambda x: (lis.count(x), x),
                reverse=True)
            return [x for x in sorted_list if len(x) > 0]

        self.memory = {smart_join(key): sort_by_cnt(vals)
                       for key, vals in self.memory.items()}
        with open(outfile, 'w') as f_handle:
            f_handle.write(json.dumps(self.memory, indent=4))

    def load_model(self, infile='models/model.json'):
        """Load in a model"""
        self.memory = defaultdict(list)
        with open(infile, 'r') as f_handle:
            dic = json.loads(f_handle.read())
        for key, value in dic.items():
            k = tuple(key.split())
            if not k:
                k = self.get_initial()
            elif len(k) == 1:
                k = tuple(list(self.get_initial())[1:] + [k[0]])
            self.memory[k] = value
