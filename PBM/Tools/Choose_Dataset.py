#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import pandas as pd
import numpy as np

def dataset(label, Hz, Moi):
    valid_labels = ['Wiley', 'JRS']
    valid_hz_values = [40, 50, 60]
    valid_moi_values = [0, 20, 40]
    
    if label not in valid_labels:
        print("Invalid label. Available labels are:", valid_labels)
        return None, None, None

    if Hz not in valid_hz_values:
        print("Invalid Hz value. Available Hz values are:", valid_hz_values)
        return None, None, None

    if Moi not in valid_moi_values:
        print("Invalid Moi value. Available Moi values are:", valid_moi_values)
        return None, None, None
    
    if label == 'Wiley':
        if Hz == 60:
            if Moi == 40:
                Input = pd.read_csv("./Data_Wiley/Wiley_60Hz_40%_input.csv")
                Output = pd.read_csv("./Data_Wiley/Wiley_60Hz_40%_output.csv")
            elif Moi == 20:
                Input = pd.read_csv("./Data_Wiley/Wiley_60Hz_20%_input.csv")
                Output = pd.read_csv("./Data_Wiley/Wiley_60Hz_20%_output.csv")
            elif Moi == 0:
                Input = pd.read_csv("./Data_Wiley/Wiley_60Hz_dry_input.csv")
                Output = pd.read_csv("./Data_Wiley/Wiley_60Hz_dry_output.csv")
        elif Hz == 40:
            if Moi == 0:
                Input = pd.read_csv("./Data_Wiley/Wiley_40Hz_dry_input.csv")
                Output = pd.read_csv("./Data_Wiley/Wiley_40Hz_dry_output.csv")
        else:
            print("No such data")
            return None, None, None
        
        radius = 0.02188
        v = 2*np.pi * radius * Hz
        
    elif label == 'JRS':
        if Hz == 60:
            if Moi == 40:
                Input = pd.read_csv("./Data_JRS/JRS_60Hz_40%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_60Hz_40%_output.csv")
            elif Moi == 20:
                Input = pd.read_csv("./Data_JRS/JRS_60Hz_20%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_60Hz_20%_output.csv")
        elif Hz == 50:
            if Moi == 40:
                Input = pd.read_csv("./Data_JRS/JRS_50Hz_40%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_50Hz_40%_output.csv")
            elif Moi == 20:
                Input = pd.read_csv("./Data_JRS/JRS_50Hz_20%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_50Hz_20%_output.csv")
        elif Hz == 40:
            if Moi == 40:
                Input = pd.read_csv("./Data_JRS/JRS_40Hz_40%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_40Hz_40%_output.csv")
            elif Moi == 20:
                Input = pd.read_csv("./Data_JRS/JRS_40Hz_20%_input.csv")
                Output = pd.read_csv("./Data_JRS/JRS_40Hz_20%_output.csv")
        else:
             print("No such data")
             return None, None, None
            
        radius = 0.02600
        v = 2*np.pi * radius * Hz
        
        
    return Input, Output, v
