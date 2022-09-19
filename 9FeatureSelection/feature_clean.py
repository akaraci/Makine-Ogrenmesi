# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:48:30 2022

@author: karaci
"""

import pandas as pd
import numpy as np
df = pd.DataFrame({'group':list('aaaab'), 
                   'val':[1,3,3,np.NaN,5],
                   'id_group':[1,np.NaN,np.NaN,np.NaN,np.NaN]})
df2 = df.loc[:, df.isnull().mean() < .8]
print(df2)
