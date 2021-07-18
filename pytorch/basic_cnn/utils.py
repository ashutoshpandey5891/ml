#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:30:32 2020

@author: t1
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def get_mnist_data(training=True,batch_size=128,shuffle = True):
    dataset = torchvision.datasets.MNIST(
        root = '../data',
        train = training,
        transform = transforms.ToTensor(),
        download = True
    )
    print('loading dataset : ','training' if training else 'test')
    print('dataset shape : ',dataset.data.shape , 'targets : ',dataset.targets.shape)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader


def get_fashion_mnist_data(training=True,batch_size=128,shuffle = True):
    dataset = torchvision.datasets.FashionMNIST(
        root = '../data',
        train = training,
        transform = transforms.ToTensor(),
        download = True
    )
    print('loading dataset : ','training' if training else 'test')
    print('dataset shape : ',dataset.data.shape , 'targets : ',dataset.targets.shape)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader

def get_cifar10_data(training=True,batch_size=128,shuffle = True,trms = None):
    
    dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = training,
        transform = transforms.ToTensor() if not trms else trms,
        download = True
    )
    print('loading dataset : ','training' if training else 'test')
    # print('dataset shape : ',dataset.data.shape , 'targets : ',dataset.targets.shape)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader

# class CNN2(nn.Module):
#     def __init__(self,NUM_CLASSES):
#         super(CNN2,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size = 3,stride = 2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3,stride = 2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size = 3,stride = 2)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.fc1 = nn.Linear(128*3*3,512)
#         self.fc2 = nn.Linear(512,NUM_CLASSES)
        
#     def forward(self,X):
#         X = F.relu(self.bn1(self.conv1(X)))
#         X = F.relu(self.bn2(self.conv2(X)))
#         X = F.relu(self.bn3(self.conv3(X)))
#         X = X.view(-1,128*3*3)
#         X = F.dropout(X,p=0.5)
#         X = F.relu(self.fc1(X))
#         X = F.dropout(X,p=0.5)
#         X = self.fc2(X)
#         return X

class CNN(nn.Module):
    def __init__(self,INPUT_CHANNELS,NUM_CLASSES):
        super(CNN,self).__init__()
        
        ## conv_block 1 output size : 32 x 16 x 16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels = INPUT_CHANNELS, out_channels = 32, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        
        ## conv block2 output size : 64 x 8 x 8
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        
        ## conv block2 output size : 128 x 4 x 4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*4*4,512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512,10)
        
    def forward(self,X):
        out = self.conv_block1(X)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = out.view(out.size(0),-1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
        

# class CNN(nn.Module):
#     def __init__(self,NUM_CLASSES):
#         super(CNN,self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3 ,stride = 2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3 , stride = 2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3 , stride = 2),
#             nn.ReLU()
#         )
        
#         self.dense_layers = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(in_features = 2*2*128, out_features = 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512,NUM_CLASSES)
#         )
        
#     def forward(self,X):
#         out = self.conv_layers(X)
#         out = out.view(out.size(0),-1)
#         out = self.dense_layers(out)
#         return out
    
