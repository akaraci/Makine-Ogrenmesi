# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 08:25:06 2022

@author: karaci
"""

from scipy.spatial import distance

def jaccard_set(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print("Intersection",intersection)
    union = (len(list1) + len(list2)) - intersection
    print("Union",union)
    return float(intersection) / union

# Define two sets 
a = [0, 1, 2, 5, 6]
b = [0, 2, 3, 4, 5, 7, 9]  #TP=3   FP=

# Find Jaccard Similarity between the two sets 
print("Jackard=",jaccard_set(a, b))