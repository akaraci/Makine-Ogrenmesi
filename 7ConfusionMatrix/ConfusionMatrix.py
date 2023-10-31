# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:49:32 2023

@author: akara
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

y_test=np.array([1,0,1,0,1,0,1])
y_pred=np.array([1,1,1,0,0,0,1])
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

sns.set(font_scale=3) #x ve y etiket font boyutunu ayarlar
ax = sns.heatmap(cf_matrix, annot=True, cmap='viridis', annot_kws={"fontsize":100})

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

