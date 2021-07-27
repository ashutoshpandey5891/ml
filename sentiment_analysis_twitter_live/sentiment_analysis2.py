#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
import pickle
import random
from nltk.classify import SklearnClassifier,ClassifierI
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier

f = open('features2.pickle','rb')
features = pickle.load(f)
f.close()

print len(features)

#randomly shuffle the features
random.shuffle(features)

#splitting into training and testing sets
train_set = features[:5000]
test_set = features[10000:]

#print len(train_set),len(test_set)
import nltk

nltk_nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
print "NLTK NB classifier score : ",nltk.classify.accuracy(nltk_nb_classifier,test_set)*100.0


mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(train_set)
print "mnb_classfier score : ",nltk.classify.accuracy(mnb_classifier,test_set)*100.0

bnb_classifier = SklearnClassifier(BernoulliNB())
bnb_classifier.train(train_set)
print "bnb_classfier score : ",nltk.classify.accuracy(bnb_classifier,test_set)*100.0

svc = SklearnClassifier(SVC(kernel = 'rbf'))
svc.train(train_set)
print "SVC : ",nltk.classify.accuracy(svc,test_set)*100.0

lin_svc = SklearnClassifier(LinearSVC())
lin_svc.train(train_set)
print "Linear SCV : ",nltk.classify.accuracy(lin_svc,test_set)*100.0

nu_svc = SklearnClassifier(NuSVC())
nu_svc.train(train_set)
print "Nu SCV : ",nltk.classify.accuracy(nu_svc,test_set)*100.0

log_reg = SklearnClassifier(LogisticRegression())
log_reg.train(train_set)
print "Logistic regression : ",nltk.classify.accuracy(log_reg,test_set)*100.0

sgd = SklearnClassifier(SGDClassifier())
sgd.train(train_set)
print "SGD : ",nltk.classify.accuracy(sgd,test_set)*100.0


class VoteClassifier(ClassifierI):
    
    def __init__(self,*classifiers):
        self.classifiers_ = classifiers
        
    def classify(self,features):
            votes = []
            for c in self.classifiers_:
                v = c.classify(features)
                votes.append(v)
            if votes.count(0) > votes.count(1):
                return 0
            elif votes.count(0) == votes.count(1):
                return self.classifiers_[0].classify(features)
            else:
                return 1
    
    def confidence(self,features):
        vote = []
        for c in self.classifiers_:
            v = c.classify(features)
            vote.append(v)
        choice = self.classify(features)
        conf = vote[vote == choice]/len(vote)
        return conf
    
vote_classifier = VoteClassifier(nltk_nb_classifier,
                                 mnb_classifier, 
                                 bnb_classifier,
                                 lin_svc,nu_svc,
                                 log_reg,sgd) 

print "Vote classifier : ",nltk.classify.accuracy(vote_classifier,test_set)*100.0

#pickle all the classifiers

classifier_list = [nltk_nb_classifier,
                   mnb_classifier, 
                   bnb_classifier,
                   lin_svc,nu_svc,
                   log_reg,sgd]

f = open('classifiers.pickle','wb')
pickle.dump(classifier_list,f)
f.close()
