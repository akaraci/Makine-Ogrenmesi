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


def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice  

imorginal = cv2.imread('hands1.jpg')

#imgray = cv2.cvtColor(imorginal, cv2.COLOR_BGR2GRAY)


#resim üzerinde active contour işlemi yapıldığında aşağıdaki resim elde edilir.
imsegmentationed = cv2.imread('segmentationed.jpg')
#imsegmentationed ile aynı boyuta getiriliyor. 
#Resimler orginal olmadığı ekran yakalama ile alındığında bu gerekli
imorginal=imorginal[:,0:397,:] 
similarity = DICE_COE(imsegmentationed,imorginal);
print("Dice=",similarity)
#cv2.imshow('Original image',orginal)
#cv2.imshow('Gray image', gray)

plt.subplot(2, 2, 1)
plt.imshow(imorginal)
plt.subplot(2, 2, 2)
plt.imshow(imsegmentationed)

plt.show()









