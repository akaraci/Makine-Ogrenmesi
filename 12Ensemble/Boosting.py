# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:06:39 2022

@author: karaci
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

X, y = load_breast_cancer(return_X_y=True)
#y=y.reshape(y.shape[0],1)
#ohe = OneHotEncoder()
#transformed = ohe.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(X_train, y_train)
result=model.score(X_test,y_test)
print("Gradient Boost Doğruluk (test_seti): ", round(result,2))


model=xgb.XGBClassifier(n_estimators=20,random_state=1,learning_rate=0.03)
model.fit(X_train, y_train)
result=model.score(X_test,y_test)
print("XG-Boost Boost Doğruluk (test_seti): ", round(result,2))