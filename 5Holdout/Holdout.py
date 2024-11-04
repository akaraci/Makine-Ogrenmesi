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

print(data.head())
X=data.iloc[:,0:-1]  #son sütun hariç diğer sütunları alır
Y=data.iloc[:,-1]   #son sütunu alır
#to_categorical, çıkışı ikilik düzende kategorik olarak kodlar (One-hot kodlama). 
#Sınıfları 0 ve 1 olarak güncelleyelim (çünkü num_classes=2)
#Sınıflar 1 ve 2 olarak kalırsa ve num_classes=2 olursa hata alırız.
#Fakat, num_classes vermezsek otomatik 3 sınıfa ayırır.Bu durum, fonksiyonun sınıf sayısını belirlerken 
#etiketteki en yüksek değeri (max sınıf + 1) alarak çalışmasından kaynaklanır.
Y = [label - 1 for label in Y]  # y = [0, 1, 0, 1]
Y = np_utils.to_categorical(Y,num_classes=2) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=30)

#stratify=Y, eğitim ve test setlerinde sınıf dağılımlarının orijinal veri setindeki sınıf dağılımıyla 
#aynı olmasını sağlar. Dengesiz veri setlerinde kullanışlıdır. 
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=30, stratify=Y)

print("All Data:",data.shape)
print("Train Data:",X_train.shape)
print("Test Data:",X_test.shape)
print("Train out Data:",Y_train.shape)
print("Test out Data:",Y_test.shape)




