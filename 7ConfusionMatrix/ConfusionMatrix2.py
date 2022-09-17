# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:21:40 2022

@author: karaci
"""


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

y_test=np.array([1,0,1,0,1,0,1])
y_pred=np.array([1,1,1,0,0,0,1])
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cf_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()