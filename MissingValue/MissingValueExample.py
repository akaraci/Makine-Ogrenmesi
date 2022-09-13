# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:47:38 2022

@author: karaci
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data=pd.read_csv('data/bikedetails.csv')
print(data.info())
print(data.isnull().sum())
print(data.shape)
print("-------------------------")
data.dropna(inplace=True)
print(data.info())
print(data.isnull().sum())
print(data.shape)

#--------mean imputer-----------------------------
#---------Method-1
meandata=pd.read_csv('data/bikedetails.csv')
print(meandata["ex_showroom_price"][:20])
meandata["ex_showroom_price"]=meandata["ex_showroom_price"].replace(np.NAN,meandata["ex_showroom_price"].mean())
print(meandata["ex_showroom_price"][:20])

#--------Method-2
meandata=pd.read_csv('data/bikedetails.csv')
print(meandata["ex_showroom_price"][:20])
fea_transformer = SimpleImputer(strategy="mean")
values = fea_transformer.fit_transform(meandata[["ex_showroom_price"]])
meandata["ex_showroom_price"]=pd.DataFrame(values)
print(meandata["ex_showroom_price"][:20])

#------------------------------------

#--------median imputer-----------------------------
#---------Method-1
mediandata=pd.read_csv('data/bikedetails.csv')
print(mediandata["ex_showroom_price"][:20])
mediandata["ex_showroom_price"]=mediandata["ex_showroom_price"].replace(np.NAN,mediandata["ex_showroom_price"].median())
print(mediandata["ex_showroom_price"][:20])

#--------Method-2
mediandata=pd.read_csv('data/bikedetails.csv')
print(mediandata["ex_showroom_price"][:20])
fea_transformer = SimpleImputer(strategy="median")
values = fea_transformer.fit_transform(mediandata[["ex_showroom_price"]])
mediandata["ex_showroom_price"]=pd.DataFrame(values)
print(mediandata["ex_showroom_price"][:20])
#------------------------------------


#----------KNN Imputer------------
from sklearn.impute import KNNImputer
# I specify the nearest neighbor to be 3 
knndata=pd.read_csv('data/bikedetails.csv')
print(knndata["ex_showroom_price"][:20])
fea_transformer = KNNImputer(n_neighbors=3)
values = fea_transformer.fit_transform(knndata[["ex_showroom_price"]])
knndata["ex_showroom_price"]=pd.DataFrame(values)
print(knndata["ex_showroom_price"][:20])


