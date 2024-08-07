#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import pandas as pd
import numpy as np

def dataset(num):
    num = int(num)
    valid_num = list(range(1, 5))
    if num not in valid_num:
        print("Invalid data. Available data num is within:", valid_num)
        return None
    
    if num == 1:
        Input = pd.read_csv("./Dataset/OriData/ori_input(black).csv")
        Output = pd.read_csv("./Dataset/OriData/ori_output(red).csv")
        duration = 0.5
    elif num == 2:
        Input = pd.read_csv("./Dataset/OriData/ori_input(black).csv")
        Output = pd.read_csv("./Dataset/OriData/ori_output(blue).csv")    
        duration = 1
    elif num == 3:
        Input = pd.read_csv("./Dataset/OriData/ori_input(black).csv")
        Output = pd.read_csv("./Dataset/OriData/ori_output(pink).csv")
        duration = 1.5
    elif num == 4:
        Input = pd.read_csv("./Dataset/OriData/ori_input(black).csv")
        Output = pd.read_csv("./Dataset/OriData/ori_output(green).csv")
        duration = 2
    
    v = 9.8
    screen = 401
    #moi = 0
    
    return Input, Output, v, screen, duration
