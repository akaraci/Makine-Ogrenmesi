# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 21:29:18 2022

@author: karaci
"""
import pandas as pd
def veriOku(file_path):
    dataset=pd.read_csv(file_path, comment='#')
    columns=dataset.columns
    for c in columns:
        print (c)
        datalar=dataset.values
        print (datalar.shape[0],datalar.shape[1])
        y=datalar[:,datalar.shape[1]-1]
        print (y)
        X=datalar[:,0:datalar.shape[1]-1]
        print (X.shape)

veriOku("./data/breastcancer.csv")