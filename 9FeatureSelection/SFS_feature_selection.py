# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:31:03 2022

@author: karaci
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB

cancer= load_breast_cancer()
can=pd.DataFrame(cancer.data, columns=cancer.feature_names)
X, y = load_breast_cancer(return_X_y=True)


# Sequential Forward Selection(sfs)
#RandomForestClassifier(n_estimators=5), LogisticRegression(), GaussianNB gibi farklı sınıflandırcılar kullanılabilir
sfs = SFS(RandomForestClassifier(n_estimators=5), 
           k_features=12,
           forward=True, #false yapılırsa backward olur
           floating=False,
           scoring = 'accuracy',
           cv = 0)
#fitting
result=sfs.fit(X, y)
#plot figure
fig1 = plot_sfs(result.get_metric_dict(), kind='std_dev')
#elde edilen sonuçlar dictionary veri tipinde elde ediliyor ve dataframe'e çevriliyor
result_LR = pd.DataFrame.from_dict(result.get_metric_dict(confidence_interval=0.90)).T
#Sonuçlar ortalama skora göre küçükten büyüğe çevriliyor
result_LR.sort_values('avg_score', ascending=0, inplace=True)
print (result_LR.head())

##1.satırdaki alfeature_idx değerini al
best_features_LR = np.array(result_LR.feature_idx.head(1).tolist()) 
#indeksi belirlenen feature'ların label'ları alınıyor
select_features_Label = can.columns[best_features_LR[0]] 
print(select_features_Label)

selected_cancer_data=can.filter(items=select_features_Label)
print(selected_cancer_data.head())