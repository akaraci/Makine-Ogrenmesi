# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:00:50 2022

@author: karaci
"""
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def verisayisi(y):
    pozitif,negatif=0,0
    for tuty in y:
        if tuty==1:
            pozitif+=1
        elif tuty==2:
            negatif+=1
    return pozitif,negatif


data=pd.read_csv("breastcancer.csv")
"""data=np.array(data)
X=data[:,0:9]
Y=data[:,9:10]"""
print(data.head())
X=data.iloc[:,0:-1]  #son sütun hariç diğer sütunları alır
Y=data.iloc[:,-1]   #son sütunu alır
#Y = np_utils.to_categorical(Y) #kategorik yapılırsa ikilik sayı sitemine kodlar

print("All Data:",data.shape)
plt.figure(figsize=(10,20))
fold_index=1;
kf = KFold(n_splits=5, shuffle=True, random_state=42)
barx=["Pozitif","Negatif"]
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]   
    print("\n-------Fold:",fold_index)
    print("Train Data:",X_train.shape)
    print("Test Data:",X_test.shape)
    fold_index+=1
    pozitif,negatif=verisayisi(Y_train)
    print("Train Pozitif data:",pozitif," Train Negatif data:",negatif)
    pozitif,negatif=verisayisi(Y_test)
    print("Test Pozitif data:",pozitif," Test Negatif data:",negatif)
    kacinci=509+fold_index #510 5 row 1 col fold_index. plot
    """plt.subplot(kacinci)
    plt.plot(Y_train,color = "red")"""
    df=[pozitif,negatif]
    plt.subplot(kacinci)
    plt.title("Fold"+str(fold_index-1))
    plt.bar(barx,df)
    




