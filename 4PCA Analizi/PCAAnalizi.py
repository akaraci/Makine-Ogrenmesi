# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:50:14 2022

@author: karaci
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

iris=load_iris()
data=iris.data
print(type(data))
feature_names=iris.feature_names
y=iris.target

df=pd.DataFrame(data,columns=feature_names)
df["sınıf"]=y
x=data
print(df.head())

fig, axs = plt.subplots(2, 2)
row,col=0,0

#1-4 feature deneniyor. Amaç boyut indirgeme olduğunda n_component=2 en uygunu
for i in range(1,5):
    pca=PCA(n_components=(i),whiten=(False))  #whiten->normalize
    x_pca=pca.fit(x)
    print("\nn_components=",i)
    print("Variance ratio=",pca.explained_variance_ratio_)
    print("Sum ratio=",sum(pca.explained_variance_ratio_))

        
    if (i>1):      
        axs[row,col].set(xlabel='Number of Component', ylabel='Cumulative Variance')
        axs[row,col].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
        axs[row,col].plot(np.cumsum(pca.explained_variance_ratio_)) #cumsum kümülatif toplamı bulur. Her bileşen için toplama ekler
        col+=1
        if (col==2):
            row+=1
            col=0
    
    if (i==4):
        df_sns=pd.DataFrame({'variance':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3','PC4']})
        sns.barplot(x='PC',y='variance',data=df_sns,color="c")
        plt.show()
    


