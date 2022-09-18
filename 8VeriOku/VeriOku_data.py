# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 21:42:02 2022

@author: karaci
"""
f=open("./data/iris.data")
turler=[]
for i,row in enumerate(f.readlines()):
    veri=row.split(",")
    if (len(veri)==5):
        if veri[4] not in turler: 
            turler.append(veri[4]) 
            f.close()
        
#f = open('./data/iris.data')
print ("cicek turLeri:",turler)
for etiket in turler:
    print (etiket)
    veriSayisi=0
    f = open("./data/iris.data") 
    for row in f.readlines(): 
        veri=row.split(",")
        if (len(veri)==5):
            if veri[4]==etiket:
                veriSayisi+=1
    print (etiket,veriSayisi) 
    f.close()
