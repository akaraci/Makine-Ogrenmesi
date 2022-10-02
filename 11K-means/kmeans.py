# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:40:52 2022

@author: karaci
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import datasets


iris =datasets.load_iris()
X=iris.data
y=iris.target

numofcluster=2
k_means_iris=KMeans(n_clusters=numofcluster, max_iter=100,  n_init=1,random_state=52)
k_means_iris.fit(X)
labels=k_means_iris.labels_

y_kmeans = k_means_iris.fit_predict(X)
centers = k_means_iris.cluster_centers_
fig, (ax0,ax1)=plt.subplots(nrows=2,figsize=(10,10))

#Verisetinde 0 ve 1. sütun saçılım grafiği çıkış 0 (y=0) ve çıkış 1 (y=1) için çizdiriliyor
ax0.scatter(X[y==0, 0], X[y==0, 1],marker='^', label ='Cluster 1')
ax0.scatter(X[y==1, 0], X[y==1, 1],marker='*',  label ='Cluster 2')
ax0.legend()


##Kmeans'den sonra verisetinde 0 ve 1. sütun saçılım grafiği tahmin çıkış 0 (y_kmeans=0) 
#ve tahmin çıkış 1 (y_kmeans=1) için çizdiriliyor
ax1.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1],marker='^', label ='Cluster 1')
ax1.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1],marker='*',  label ='Cluster 2')

#Küme merkez noktaları grafiğe yansıtılıyor
ax1.scatter(centers[:, 0], centers[:, 1], s=100, c='red')   
ax1.legend()


from sklearn.metrics import confusion_matrix
from sklearn import metrics
# cm=confusion_matrix(y, y_kmeans)
# print(cm)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
# cm_display.plot()
# plt.show()


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_kmeans,y) #çok doğru sonuç vermez. çünkü gerçek veri setinde çıkış 2'de var.
print("Accuracy=",acc)