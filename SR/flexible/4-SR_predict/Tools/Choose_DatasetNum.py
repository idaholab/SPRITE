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
    valid_num = list(range(1, 17))
    if num not in valid_num:
        print("Invalid data. Available data num is within:", valid_num)
        return None
    
    if num == 1:
        Hz = 40
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_40Hz_20%_0.5_output.csv")
    elif num == 2:
        Hz = 40
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_40Hz_20%_0.75_output.csv")    
    elif num == 3:
        Hz = 40
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_40Hz_40%_0.5_output.csv")
    elif num == 4:
        Hz = 40
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_40Hz_40%_0.75_output.csv")
    elif num == 5:
        Hz = 50
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_50Hz_20%_0.5_output.csv")
    elif num == 6:
        Hz = 50
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_50Hz_20%_0.75_output.csv")
    elif num == 7:    
        Hz = 50
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_50Hz_40%_0.5_output.csv")
    elif num == 8:    
        Hz = 50
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_50Hz_40%_0.75_output.csv")
    elif num == 9:
        Hz = 60
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_60Hz_20%_0.5_output.csv")
    elif num == 10:
        Hz = 60
        moi = 20
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_60Hz_20%_0.75_output.csv")
    elif num == 11:
        Hz = 60
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_60Hz_40%_0.5_output.csv")
    elif num == 12:
        Hz = 60
        moi = 40
        Input = pd.read_csv("./Dataset/Data_JRS/JRS_input.csv")
        Output = pd.read_csv("./Dataset/Data_JRS/JRS_60Hz_40%_0.75_output.csv")
    elif num == 13:   
        Hz = 40
        moi = 15
        Input = pd.read_csv("./Dataset/Data_Wiley/Wiley_input.csv")
        Output = pd.read_csv("./Dataset/Data_Wiley/Wiley_40Hz_dry_output.csv")
    elif num == 14:
        Hz = 60
        moi = 15
        Input = pd.read_csv("./Dataset/Data_Wiley/Wiley_input.csv")
        Output = pd.read_csv("./Dataset/Data_Wiley/Wiley_60Hz_dry_output.csv")
    elif num == 15:
        Hz = 60
        moi = 20
        Input = pd.read_csv("./Dataset/Data_Wiley/Wiley_input.csv")
        Output = pd.read_csv("./Dataset/Data_Wiley/Wiley_60Hz_20%_output.csv")
    elif num == 16:
        Hz = 60
        moi = 40
        Input = pd.read_csv("./Dataset/Data_Wiley/Wiley_input.csv")
        Output = pd.read_csv("./Dataset/Data_Wiley/Wiley_60Hz_40%_output.csv")
    
        
    if num in list([13,14,15,16]):
        screen = 10
        radius = 0.02188
        v = 2*np.pi * radius * Hz
    elif num in list([1,3,5,7,9,11]):
        screen = 12.7
        radius = 0.02600
        v = 2*np.pi * radius * Hz
    else:
        screen = 19.05
        radius = 0.02600
        v = 2*np.pi * radius * Hz
    return Input, Output, v, screen, moi
