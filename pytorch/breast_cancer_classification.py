#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:39:54 2020

@author: t1
"""
import sys,os,getopt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main(plot_data = False):
    data = load_breast_cancer()
    print(data.target_names)
    print('data : ',data.data.shape,'targets : ',data.target.shape)
    N,D = data.data.shape
    # split train test
    X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.33,random_state = 101)
    print('Train : ',X_train.shape,Y_train.shape)
    print('Test : ',X_test.shape,Y_test.shape)
    
    # normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # model
    model = nn.Sequential(
            nn.Linear(D,1),
            nn.Sigmoid()
        )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # data preparation
    X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))
    Y_train_tensor = torch.from_numpy(Y_train.reshape(-1,1).astype(np.float32))
    Y_test_tensor = torch.from_numpy(Y_test.reshape(-1,1).astype(np.float32))
    
    # training the model
    n_epochs = 500
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(n_epochs):
        
        optimizer.zero_grad()
        # forward pass
        output = model(X_train_tensor)
        loss = criterion(output,Y_train_tensor)
        
        # compute gradients
        loss.backward()
        optimizer.step()
        
        # compte test scores
        output_test = model(X_test_tensor)
        test_loss = criterion(output_test,Y_test_tensor)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        # compute accuracies
        with torch.no_grad():
            
            output_train = model(X_train_tensor).numpy()
            output_train = (output_train >= 0.5)
            train_acc = (output_train == Y_train.reshape(-1,1)).mean()
            
            output_test = model(X_test_tensor).numpy()
            output_test = (output_test >= 0.5)
            test_acc = (output_test == Y_test.reshape(-1,1)).mean()
            # print(output_test)
            
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        print(f'epoch : {epoch}/{n_epochs} train_loss : {loss.item():.4f} test_loss : {test_loss.item():.4f}')
    
    if plot_data:
        plt.plot(train_losses,label = 'train_label')
        plt.plot(test_losses,label = 'test_label')
        plt.title('Losses')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(train_accs,label = 'train acc')
        plt.plot(test_accs,label = 'test acc')
        plt.title('accuracies')
        plt.legend()
        plt.show()
    
    # computing train test accuracies
    with torch.no_grad():
        train_out = (model(X_train_tensor).numpy() > 0.5)
        test_out = (model(X_test_tensor).numpy() > 0.5 )
        
        
    train_acc = (train_out == Y_train.reshape(-1,1)).mean()
    test_acc = (test_out == Y_test.reshape(-1,1)).mean()
    print(f'train_acc : {train_acc:.4f}  test_acc : {test_acc:.4f}')
    
    ## saving the model
    torch.save(model.state_dict(),'saved_breast_cacner.pt')
    
    MODEL_PATH = 'saved_breast_cacner.pt'
    
    print('loading trined model : {}'.format(MODEL_PATH))
    model2 = nn.Sequential(
            nn.Linear(D,1),
            nn.Sigmoid()
        )
    
    model2.load_state_dict(torch.load(MODEL_PATH))
    with torch.no_grad():
        train_out = (model(X_train_tensor).numpy() > 0.5)
        test_out = (model(X_test_tensor).numpy() > 0.5 )
        
        
    train_acc = (train_out == Y_train.reshape(-1,1)).mean()
    test_acc = (test_out == Y_test.reshape(-1,1)).mean()
    print(f'train_acc : {train_acc:.4f}  test_acc : {test_acc:.4f}')
    
    

if __name__ == '__main__':
    args = sys.argv
    plot_data = False
    try : 
        arg = args[1]
        if arg == 'true':
            plot_data = True
    except Exception as e:
        print('no arguments given : ',str(e))
    main(plot_data)
