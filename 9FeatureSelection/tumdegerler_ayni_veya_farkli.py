# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:07:20 2022

@author: karaci
"""
import pandas as pd
df = pd.DataFrame({
'SözleşmeID': [101, 102, 103, 104, 105, 106, 107, 108],
'MüşteriSüre': [10, 11, 12, 10, 11, 10, 12, 10],
'Şehir': ['Bursa', 'Bursa', 'Kocaeli', 'Bursa', 'Bursa', 'Bursa', 'Bursa', 'Bursa'],
'FaturaTutar': ['100', '120', '110', '90', '80', '150', '45', '65']
})
print(df.dtypes)
print(df.nunique())
print(df.shape)

uniqnumber=df.nunique()
df2=df.loc[:,uniqnumber==df.shape[0]] #tüm değerleri farklı olanları al
df3=df.loc[:,uniqnumber<=3] #3 ve daha az çeşit veri olanları al

