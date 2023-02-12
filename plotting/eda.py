import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
impor yellowbrick

## Radviz
from yellowbrick.features import RadViz

def plot_numerical_features(train_data,num_cols,target_col=None):
    n = len(num_cols)
    fig,axes = plt.subplots(n,3,figsize=(22,n*4))
    for i in range(n):
        col = num_cols[i]
    #     sns.histplot(data=train_data,x=col,hue=target_col,ax=axes[i][0])
    #     sns.boxplot(data = train_data,x=target_col,y=col,ax=axes[i][1])
        sns.histplot(data=train_data,x=col,ax=axes[i][0])
        sns.boxplot(data = train_data,y=col,ax=axes[i][1])
        sns.scatterplot(data = train_data,x=col,y=target_col,ax=axes[i][2])

def plot_heatmap(train_data,num_cols):
    corr = train_data[num_cols].corr()
    fig = plt.figure(figsize=(16,8))
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='Blues')
    plt.xticks(rotation=45);


## RadViz plot , more explain at : https://orange3.readthedocs.io/projects/orange-visual-programming/en/latest/widgets/visualize/radviz.html
## yellowbrick documnetation : https://www.scikit-yb.org/en/latest/api/features/radviz.html
def plot_radviz(X,y):
    classes = list(y.unique())
    fig = plt.figure(figsize=(12,8))
    vizer = RadViz(classes=classes)
    vizer.fit(X,y)
    vizer.transform(X)
    vizer.show()


def plot_confusion_matrix(y_test,preds,labels):
    mat = confusion_matrix(y_test,preds,labels = labels)
    mat_df = pd.DataFrame(mat,index=labels,columns=labels)
    _ = plt.figure(figsize=(16,12))
    sns.heatmap(mat_df,annot=True,fmt='.2f',cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

