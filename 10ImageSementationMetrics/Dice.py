# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:03:34 2022

@author: karaci
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian

from sklearn.metrics import jaccard_score

def jackard(y_true, y_pred):
    jaccard = jaccard_score(y_true.flatten(), y_pred.flatten(),average="macro")
    return jaccard


# def dice_coefficient(set1, set2):
#     intersection =len(np.intersect1d(set1,set2)) #len(set1.intersection(set2))
#     union = len(set1) + len(set2)
#     if union == 0:
#         return 0  # Bölme hatası önleme
#     print(intersection)
#     print(union)
#     dice_score = (2.0 * intersection) / union
#     return dice_score


def dice_coefficient(y_true, y_pred):
    jaccard = jaccard_score(y_true.flatten(), y_pred.flatten(),average="macro")
    return 2*jaccard / (1 + jaccard)


imorginal = cv2.imread('hands1.JPG')

#imgray = cv2.cvtColor(imorginal, cv2.COLOR_BGR2GRAY)

#resim üzerinde active contour işlemi yapıldığında aşağıdaki resim elde edilir.
imsegmentationed = cv2.imread('segmentationed.jpg')
#imsegmentationed ile aynı boyuta getiriliyor. 
#Resimler orginal olmadığı ekran yakalama ile alındığında bu gerekli
imorginal=imorginal[:,0:397,:] 

dice=dice_coefficient(imsegmentationed,imorginal)
print("Dice=",dice)

jackard=jackard(imsegmentationed,imorginal)
print("Jackard=",jackard)

dice=dice_coefficient(imorginal,imorginal)
print("Dice=",dice)


plt.subplot(2, 2, 1)
plt.imshow(imorginal)
plt.subplot(2, 2, 2)
plt.imshow(imsegmentationed)

plt.show()









