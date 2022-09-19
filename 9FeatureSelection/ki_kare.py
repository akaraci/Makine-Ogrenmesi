# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:36:45 2022

@author: karaci
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
iris_dataset = load_iris()
#Girdi ve Çıktı Değişkeni
X = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names)
y = iris_dataset.target

#Ki kare testine göre en iyi sonuç veren 2 özelliği seç
chi2_features = SelectKBest(chi2, k = 2) #özellikleri belirle
X_kbest_features = chi2_features.fit_transform(X, y) #özellikleri seç ve yeni veri seti oluştur

#özellik etiketlerini bul ve yaz
print(chi2_features.get_support()) #array([False, False,  True,  True]) çıktısını verir
features = X.columns[chi2_features.get_support()]
print("Seçilen Özellikler:",features)

#Azaltılmış Özellikler
print('Orjinal Özellik Sayısı:', X.shape[1])
print('Seçilmiş Özellik Sayısı:', X_kbest_features.shape[1])

