#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:37:03 2020

@author: t1
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 20
D = 1
X = np.random.random(N)*10 - 5
Y = 0.5*X + 1 + np.random.randn(N)
# plt.scatter(X,Y)
# plt.show()

## pytorch model training process
# create model
# create creiterion(loss)  : eg -> criterion = nn.MSELoss()
# create optimizer : eg -> torch.optim.SGD(model.parameters(),lr)
# prepare data (appropriate dims and convert to torch tensor)
# train the model 

## pytorch linear model

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

X = X.reshape(N,D)
Y = Y.reshape(N,1)

input_ = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

# print(type(input_),type(output_))

# training loop
n_epochs = 30
losses = []
for epoch in range(n_epochs):
    
    # remove accumulated gradients
    optimizer.zero_grad()
    
    # forward pass
    output_ = model(input_)
    loss = criterion(output_,targets)
    losses.append(loss.item())
    # compute gradients
    loss.backward()
    optimizer.step()
    
    print(f'epoch : {epoch+1}/{n_epochs} loss : {loss.item():.4f}')

plt.plot(losses)
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss per iteration')
plt.show()    
    
# predicted = None
# get model weights
w = model.weight.data.numpy()[0][0]
b = model.bias.data.numpy()[0]
with torch.no_grad():
    predicted = model(input_).numpy()
plt.plot(X,predicted,label = 'predicted points')
plt.scatter(X,Y,label = 'actual data')
plt.title(f'W : {w:.2f} , b : {b:.2f}')
plt.legend() 
plt.show()   


    
    
    
    
    
    
