# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:50:42 2022

@author: karaci
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


modeldt = DecisionTreeClassifier()
modeldt.fit(X_train,y_train)
dt_test_sonuc = modeldt.score(X_test, y_test)
print("Karar Ağacı Doğruluk (test_seti): ",round(dt_test_sonuc,2))

baggingObject = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,  n_estimators=20)
baggingObject.fit(X_train, y_train)
baggingObject_sonuc = baggingObject.score(X_test, y_test)
print("Bagging Doğruluk (test_seti): ", round(baggingObject_sonuc,2))

