import email
from collections import Counter
from collections import defaultdict
import os
import math
import json
from math import log
from nltk.corpus import stopwords
rootdir = '/home/ems316/enron1/ham/'
hamDictIDF = Counter()
hamDictTFList = list()
tfidfList = list()
hamDict = Counter()
N = 16545
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        f = open(os.path.join(subdir, file), 'r')
        msg = email.message_from_file(f)
        payload = msg.get_payload()
        body = ''.join([x if x.isalnum() else ' ' for x in payload])
        tokens = [word.lower() for word in body.split() if word.lower() not in 
                  stopwords.words('english')]
        seen = set()
        seen_add = seen.add
        hamDictTF = Counter()
        for word in tokens:
            if not (word in seen or seen_add(word)):
                hamDictIDF[word] += 1
            hamDict[word] += 1
            hamDictTF[word] += 1
        hamDictTFList.append(hamDictTF)
    
    hamDict = hamDict.most_common(10)
    for word, _ in hamDict:
        hamDictIDF[word] = 0 if hamDictIDF[word] <= 0 else log(N*1.0/hamDictIDF[word])

    for elemnt in hamDictTFList:
        tfidf = Counter()
        for word, _ in hamDict:
            tfidf[word] = elemnt[word] * hamDictIDF[word]
        tfidfList.append(dict(tfidf)) 
with open('data.json', 'w') as fp:
    json.dump(tfidfList, fp, indent=4)
