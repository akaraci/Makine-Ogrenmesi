# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:41:05 2022

@author: karaci
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

y_test=np.array([1,0,1,0,1,0,1])
y_pred=np.array([1,1,1,0,0,0,1])
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)



ax = sns.heatmap(cf_matrix, annot=True, cmap='viridis')

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
#plt.show()