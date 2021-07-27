#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import stopwords
import re
import pickle

f = open('positive.txt','r').read().lower()
pos_rev = [(line,'pos') for line in f.split('\n')]
f = open('negative.txt','r').read().lower()
neg_rev = [(line,'neg') for line in f.split('\n')]


allowed_word_types = ['J']
all_words = []
for revs in [pos_rev,neg_rev]:
    for rev in revs:
        words = rev[0].split()
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0])
        
print len(all_words)

stop_words = set(stopwords.words('english'))
all_words = nltk.FreqDist(all_words)
feats = []

for word,count in all_words.most_common(10000):
    if len(feats) >= 3000:
        break
        
    if word not in stop_words and re.findall('[^a-z]',word) == []:
        feats.append(word)
        
print feats[:20]
print len(feats)

f = open('text_features.pickle','wb')
pickle.dump(feats,f)
f.close()

def find_feats(rev):
    rev_words = rev.split()
    features = {}
    for word in feats:
        features[word] = word in rev_words
    return features

features = [(find_feats(rev),category) for revs in [pos_rev,neg_rev] 
            for rev,category in revs]

print(len(features))
#save the features in a pickle file


f = open('features2.pickle','wb')
pickle.dump(features,f)
f.close()
