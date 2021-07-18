#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:03:36 2019

@author: t1
XGBoost 
"""
import numpy as np
import pandas as pd

#import the data
dataset = pd.read_csv('data/Churn_Modelling.csv')

labels = dataset['Exited']
features = dataset.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1)

## data preprocessing ##
features['Gender'] = features['Gender'].map({'Male':1,'Female':0})
features = pd.concat([features,pd.get_dummies(features['Geography'])],axis=1)
features = features.drop(['Geography'],axis=1)

# test train split #
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(features,labels)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

## XGBoost Model ##

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_x,train_y)
preds = model.predict(test_x)

## evaluation
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV,cross_val_score

print 'Test set results : '
print 'Accuracy : ',accuracy_score(preds,test_y)
print 'Confusion Matrix : \n',confusion_matrix(preds,test_y)

## cross validation
cross_val_accs = cross_val_score(model,train_x,train_y,cv=10)
print "Cross val mean  : ",cross_val_accs.mean()
print "Cross val Std : ",cross_val_accs.std()

##Grid search for optimal parameters
parameters = [{'max_depth' : [3,10,20],
               'learning_rate' :[0.01,0.1,1.0],
               'n_estimators' : [20,50,100]}]

grid_search = GridSearchCV(estimator=model,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(train_x,train_y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print 'grid Search best score : ',best_score