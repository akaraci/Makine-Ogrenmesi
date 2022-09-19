# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:16:01 2022

@author: karaci
"""

from sklearn.feature_selection import VarianceThreshold
X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold(0.2) #default
X1=selector.fit_transform(X)
print("X=",X)
print("X1=\n",X1)

