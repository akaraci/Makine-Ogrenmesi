# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:00:50 2022

@author: karaci
"""
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

data=pd.read_csv("breastcancer.csv")
"""data=np.array(data)
X=data[:,0:9]
Y=data[:,9:10]"""
print(data.head())
X=data.iloc[:,0:-1]  #son sütun hariç diğer sütunları alır
Y=data.iloc[:,-1]   #son sütunu alır
#Y = np_utils.to_categorical(Y) #kategorik yapılırsa ikilik sayı sitemine kodlar
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=30)
print("All Data:",data.shape)
print("Train Data:",X_train.shape)
print("Test Data:",X_test.shape)




