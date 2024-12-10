# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:23:37 2024

@author: akara
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Iris veri seti
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

df=pd.DataFrame(data=X,columns=feature_names)
df["target"]=y

# Random Forest modeli
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Özellik önemleri elde ediliyor ve dataframe'e ekleniyor.
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values("Importance", ascending=False)
print("Özellik Önemleri:\n", importance_df)

#importance 0.4'den büyük olanların featurename'i bul.
selectedimportance=importance_df.Feature[importance_df.Importance>0.4].tolist()
#sadece belirlenen feature'ları al. 
dfnew=df[selectedimportance]
dfnew["target"]=y

