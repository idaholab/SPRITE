#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Minglei Lu
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def regres(moi,v,screen):
    clf = RandomForestRegressor()
    df = pd.read_csv("./Tools/regres_data_JRS.csv")
    X = df[['moisture', 'speed', 'screen']]
    Y = df[['fmat', 'xWmmin', 'gamma', 'alpha', 'cdratio']]
    clf.fit(X, Y)
    # Wiley speed 5.5 6.9 8.2
    # JRS speed 6.5 8.2 9.8
    pred = clf.predict([[moi,v,screen]])
    x = np.array([pred[0,0], pred[0,1], pred[0,2], pred[0,3], pred[0,4]])
    return x
