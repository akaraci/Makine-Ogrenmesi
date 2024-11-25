# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:36:55 2022

@author: karaci
"""

from sklearn.datasets import fetch_california_housing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = fetch_california_housing()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MedHouseVal"] = x.target

print(df.head())
#Pearson Korelasyon
plt.figure(figsize=(12,10))
cor = df.corr() #korelasyon hesaplar
print(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Hedef değişken ile olan korelasyon. korelasonu yönü önemli 
#olmadığı şiddeti bizim için önemli o yüzden mutlak değerini alıyoruz.
cor_target = abs(cor["MedHouseVal"]) 
#Hedef değişkenle yüksek korelasyonlu olanları seç
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

