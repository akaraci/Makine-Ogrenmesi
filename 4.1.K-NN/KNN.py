# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:40:03 2022

@author: karaci
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# csv dosyamızı okuduk.
iris=load_iris()
data=iris.data

# Bağımlı Değişkeni ( species) bir değişkene atadık
species = iris.target

x_train, x_test, y_train, y_test = train_test_split(data,species,test_size=0.33,random_state=0)


# KNeighborsClassifier sınıfından bir nesne ürettik
# n_neighbors : K değeridir. Bakılacak eleman sayısıdır. Default değeri 5'tir.
# metric : Değerler arasında uzaklık hesaplama formülüdür.
#p=2 minkowski uzaklığı üs parametresi p=2 olursa öklit uzaklığı olur
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
knn.fit(x_train,y_train)

# Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık
result = knn.predict(x_test)

# Karmaşıklık matrisi
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
accuracy = accuracy_score(y_test, result)
print(accuracy)


