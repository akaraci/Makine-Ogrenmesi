# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:18:13 2022

@author: karaci
"""
import pandas as pd
import numpy as np

#%%
def to_array_and_reshape(X):
    X=np.array(X)
    X=X.reshape(X.shape[0]*X.shape[1])
    return X

def func_pandas_data_frame(X,y):
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    return X,y

def birlestir(X,y):
    X,y=func_pandas_data_frame(X,y)  
    concatX= pd.concat([X,y]) 
    concatX=np.array(concatX)
    return concatX
#%%

dataset_name ='Parkinson.xlsx'
file_path = "./data/"+dataset_name
print (file_path," isleniyor...")

dataset=pd.read_excel(file_path)  #pandas

print("\n------------Dataset top 10--------------\n")
print(dataset.head())
            
print("\n------------Dataset Value--------------\n")
veriseti=dataset.values
print (veriseti.shape)
print(veriseti)

print("\n------Columns Name-----------\n")
columns=dataset.columns
for i,c in enumerate(columns):
    if (i<10):print(c,end=",")
#%%

datasetcsv=pd.read_csv("./data/breastcancer.csv")
print("\n\n",datasetcsv.head())
datasetcsvvalues=datasetcsv.values
print(datasetcsvvalues.shape)
print(datasetcsvvalues)

Z=to_array_and_reshape(datasetcsvvalues)
print("\n\n--------------",type(Z),"  ",Z.shape)

concatDataSet=birlestir(veriseti,datasetcsvvalues)
print("Concat Dataset Shape",concatDataSet.shape)

columnsname=datasetcsv.columns
print(columnsname)

X=datasetcsv.iloc[:,0:-1]
y=datasetcsv.iloc[:,-1]

print(X.head())
print(y.head())
#%%


