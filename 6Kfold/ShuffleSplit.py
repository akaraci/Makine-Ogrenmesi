# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 21:16:31 2022

@author: karaci
"""

from sklearn.model_selection import ShuffleSplit
import numpy as np

X = np.arange(10)
ss = ShuffleSplit(n_splits=5, train_size=0.75, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))