#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import pickle
import nltk
from nltk.classify import ClassifierI

#f = open('features2.pickle','rb')
#features = pickle.load(f)
#f.close()

#test_set = features[10000:]

f = open('text_features.pickle','rb')
feature_words = pickle.load(f)
f.close()

f = open('classifiers.pickle','rb')
classifier_list = pickle.load(f)
f.close()

#classifiers
nltk_nb_classifier = classifier_list[0]
mnb_classifier = classifier_list[1]
bnb_classifier = classifier_list[2]
lin_svc = classifier_list[3]
nu_svc = classifier_list[4]
log_reg = classifier_list[5]
sgd = classifier_list[6]


class VoteClassifier(ClassifierI):
    
    def __init__(self,*classifiers):
        self.classifiers_ = classifiers
        
    def classify(self,features):
            votes = []
            for c in self.classifiers_:
                v = c.classify(features)
                votes.append(v)
            if votes.count('neg') > votes.count('pos'):
                return 'neg'
            elif votes.count('neg') == votes.count('pos'):
                return self.classifiers_[0].classify(features)
            else:
                return 'pos'
    
    def confidence(self,features):
        vote = []
        for c in self.classifiers_:
            v = c.classify(features)
            vote.append(v)
        choice = self.classify(features)
        conf = float(vote.count(choice))/len(vote)
        return conf
    
vote_classifier = VoteClassifier(nltk_nb_classifier,
                                 mnb_classifier, 
                                 bnb_classifier,
                                 lin_svc,nu_svc,
                                 log_reg,sgd)

def find_features(rev):
    rev_words = rev.split()
    features = {}
    for word in feature_words:
        features[word] = word in rev_words
    return features

def sentiment(text):
    feats = find_features(text)
    return vote_classifier.classify(feats),vote_classifier.confidence(feats)
