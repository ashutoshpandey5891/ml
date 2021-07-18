#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:29:20 2020

@author: t1
"""

import time
from utils import get_mnist_data,CNN,plot_confusion_matrix,get_fashion_mnist_data
from utils import get_cifar10_data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix

BATCH_SIZE = 128
N_EPOCHS = 20
NUM_CLASSES = 10
INPUT_CHANNELS = 3
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = 'trained_model_cifar.pt'

transforms_ = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor()
])

print('loading data ...')
# train_loader = get_mnist_data(training = True,batch_size = BATCH_SIZE,shuffle = True)
# test_loader = get_mnist_data(training = False,batch_size = BATCH_SIZE,shuffle = False)

# train_loader = get_fashion_mnist_data(training = True,batch_size = BATCH_SIZE,shuffle = True)
# test_loader = get_fashion_mnist_data(training = False,batch_size = BATCH_SIZE,shuffle = False)

train_loader = get_cifar10_data(training = True,batch_size = BATCH_SIZE,shuffle = True,trms = transforms_)
test_loader = get_cifar10_data(training = False,batch_size = BATCH_SIZE,shuffle = False)

print('creating model ...')
print('using devices : {}'.format(DEVICE))
model = CNN(INPUT_CHANNELS,NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

### training loop
train_losses = []
test_losses = []
for epoch in range(N_EPOCHS):
    print(f'starting epoch : {epoch}')
    start_time = time.clock()
    train_batch_loss = []
    for inputs,targets in train_loader:
        inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item())
    epoch_loss = np.mean(train_batch_loss)
    train_losses.append(epoch_loss)
    
    test_loss = []
    for inputs,targets in test_loader:
        inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        test_loss.append(loss.item())
    epoch_test_loss = np.mean(test_loss)
    test_losses.append(epoch_test_loss)
    time_taken = time.clock() - start_time
    print(f'Epoch : {epoch},Time : {time_taken:.4f},train loss : {epoch_loss:.4f},test loss : {epoch_test_loss:.4f}')


plt.plot(train_losses,label = 'training loss')
plt.plot(test_losses,label = 'test loss')
plt.xlabel('epochs')
plt.title('Loss plot')
plt.show()


print('saving model as : {}'.format(model_path))
torch.save(model.state_dict(),model_path)

print('computing accuracies')

n_correct = 0
n_total = 0
for inputs,targets in train_loader:
    inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
    outputs = model(inputs)
    _,preds = torch.max(outputs,1)
    n_correct += (preds == targets).sum().item()
    n_total += targets.shape[0]
train_acc = n_correct/n_total

n_correct = 0
n_total = 0
preds_all = np.array([])
targets_all = np.array([])
for inputs,targets in test_loader:
    inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
    outputs = model(inputs)
    _,preds = torch.max(outputs,1)
    n_correct += (preds == targets).sum().item()
    n_total += targets.shape[0]
    
    preds_all = np.concatenate((preds_all,preds.cpu().numpy()))
    targets_all = np.concatenate((targets_all,targets.cpu().numpy()))
test_acc = n_correct/n_total

print(f'training acc : {train_acc:.4f}, test acc : {test_acc:.4f}')
print('confusion matrix')
cm = confusion_matrix(y_true=targets_all, y_pred = preds_all)
plot_confusion_matrix(cm, classes = list(range(10)))