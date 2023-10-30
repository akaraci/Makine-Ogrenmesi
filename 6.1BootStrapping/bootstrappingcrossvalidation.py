# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:27:58 2023

@author: akara
"""

from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target
accuracies=[]
model=KNeighborsClassifier()

for i in range(30):
    X_train,y_train=resample(X,y,n_samples=int(0.75*len(X)),random_state=i)
    X_test,y_test=resample(X,y,n_samples=int(0.25*len(X)),random_state=i)
    model.fit(X_train,y_train)
    accuracy=model.score(X_test, y_test)
    accuracies.append(accuracy)
    
print("Ortalama Doğruluk",sum(accuracies)/len(accuracies))