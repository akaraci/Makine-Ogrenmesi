# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:19:27 2023

@author: akara
"""

import glob, os

# Current directory
current_dir = "D:/Origa/Labeledalldata/"
print(current_dir)


for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.txt")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(title,ext)
    file = open(current_dir+title + '.txt', 'r')
    row=file.readline();
    #print(row[0:2])
    newrow=row[2:-1]
    print(newrow)
    newrow='0'+newrow
    print(newrow)
    filenew=open(current_dir+title+'.txt',"w")
    filenew.write(newrow)
    filenew.close()
    file.close()