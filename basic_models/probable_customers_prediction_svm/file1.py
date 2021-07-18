#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:58:08 2019

@author: t1
Cross validation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the data
dataset = pd.read_csv('data/Social_Network_Ads.csv')

labels = dataset['Purchased']
features = dataset.drop(['User ID','Purchased'],axis=1)

#label encoding
features['Gender'] = features['Gender'].map({'Male':0,'Female' : 1})

#splitting between train and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(features,labels,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

#create a SVM using train data
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')
classifier.fit(train_x,train_y)

#predictions
preds = classifier.predict(test_x)
from sklearn.metrics import confusion_matrix,accuracy_score
print "Accuracy on SVM : ",accuracy_score(preds,test_y)
print "Confusion Matrix : \n",confusion_matrix(preds,test_y)


#using k fold cross validation
from sklearn.model_selection import cross_val_score
accs = cross_val_score(estimator = classifier,X =train_x,y=train_y,cv=10)
print "Cross Val Mean : ",accs.mean()
print "Cross val std : ",accs.std()
