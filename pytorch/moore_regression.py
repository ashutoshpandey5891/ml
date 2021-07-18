#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:34:51 2020

@author: t1
"""
import sys,getopt


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

def main(plot_data = False):    
    data = pd.read_csv('data/moore.csv',header = None)
    X = data[0].values.reshape(-1,1)
    Y = data[1].values.reshape(-1,1)
    
    if plot_data:
        plt.scatter(X,Y)
        plt.title(' c = c1 * r ** t')
        plt.show()
    
    Y_log = np.log(Y)
    
    if plot_data:
        plt.scatter(X,Y_log)
        plt.title('log(c) = t * log(r) + b')
        plt.show()
        
    # standardization
    mx = X.mean()
    sx = X.std()
    my = Y_log.mean()
    sy = Y_log.std()
    print('X  mean : {} std : {}'.format(mx,sx))
    print('Y  mean : {} std : {}'.format(my,sy))
    nX = (X - mx)/sx
    nY = (Y_log - my)/sy

    
    if plot_data:
        plt.scatter(nX,nY)
        plt.title('X = log(r) * Y')
        plt.show()
        
    # creating pytorch model
    model = nn.Linear(1,1)    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
    
    # data conversion
    input_ = torch.from_numpy(nX.astype(np.float32))
    targets = torch.from_numpy(nY.astype(np.float32))
    
    # training the model
    n_epochs = 100
    losses = []
    for epoch in range(n_epochs):
        
        # remove accumulated gradients
        optimizer.zero_grad()
        
        # forward pass
        output = model(input_)
        
        #compute loss and gradients
        loss = criterion(output, targets)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
        
        print(f'epoch : {epoch}/{n_epochs} loss : {loss.item():.4f}')
        
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.show()
    
    preds = None
    w = model.weight.data.numpy()[0][0]
    b = model.bias.data.numpy()[0]
    with torch.no_grad():
        preds = model(input_).numpy()
    plt.plot(nX,preds,label = 'predicted')
    plt.scatter(nX,nY,label = 'actual')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'log(nY) = {w:.4f}nX + {b:.4f}')
    plt.legend()
    plt.show()
    print(f'r = {np.exp(w*(sy/sx)):.4f}')
    print(f'time to 2 : {np.log(2)/(w*(sy/sx)):.4f}')
    

if __name__ == '__main__':
    plot_data = False
    try : 
        args = sys.argv[1:]
        opts,args = getopt.getopt(args,"hp:",["plot_data="])
    except getopt.GetoptError:
        print('python3 moore_regression.py -p <true/false>')
    for opt ,arg in opts:
        if opt == '-h':
            print('python3 moore_regression.py -p <true/false>')
        elif opt in ('-p','--plot_data'):
            if arg == 'true':
                plot_data = True
    main(plot_data)
            
            
    
