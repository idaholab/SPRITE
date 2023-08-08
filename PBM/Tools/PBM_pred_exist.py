# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 22:38:52 2023

@author: Minglei Lu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import platform
import subprocess
from Model_screen import myModel
from regression import regres
from Choose_DatasetNum import dataset

while True:
        data_num = input("Enter the No. data you want choose (choose between 1~16)\n\
\nPlease enter here: ")
        if data_num == "":
            print('\nNo input detected, using 1st dataset as default.\n')
            data_num = 1
            break
        
        try:
            data_num = int(data_num)
            if 1 <= data_num <= 16:
                print(f"\nData No.{data_num} is selected.\n")
                break
            else:
                print("\n############# Warning ############## \n\
Input out of range. Please enter an integer between 1 and 16.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input. Please enter an integer.\n\
####################################\n")
Input, Output, v, screen, moi = dataset(data_num)

FSieves = np.array(Input[['FSieves']])      #FSieves (Sieves used in feed PSD) as a vector. (PSD: partical size distribution)
Feedmass = np.array(Input[['Feedmass']])    #Feedmass (Cumulative distribution of experimental feed %mass retained) as a matrix
ExpSieves = np.array(Output[['ExpSieves']])  #ExpSieves (Sieves used in experimental PSD) as a vector
Expmass = np.array(Output[['Expmass']])      #Expmass (Cumulative distribution of experimental product %mass retained) as a matrix

x = regres(moi,v,screen)
auser_model, fuser_model, p_sieve_model, p_static_cumulative = myModel(Input, Output, v, x[0], x[1], x[2], x[3], x[4], screen=screen)

# R^2
f = interp1d(p_sieve_model, p_static_cumulative)
p_transient_cumulative_point = f(ExpSieves)
#total sum of squares (SST)
Expmass_mean = np.mean(Expmass)
SST = np.sum(np.square(Expmass-Expmass_mean))
#sum squared regression (SSR)
SSR = np.sum(np.square(p_transient_cumulative_point-Expmass))
#R^2
R2 = 1-SSR/SST
print('Coefficient of determination (R^2) = ', R2)

plt.plot(FSieves, Feedmass*100, '.', label="Feed PSD")
plt.plot(ExpSieves, Expmass*100, 'b^', label="Expirenment")
plt.plot(p_sieve_model, p_static_cumulative*100, 'r-', label="Model pred")
plt.xlabel('Sieve size (mm)')
plt.ylabel('Cumulative product PSD (%)')
plt.legend()
plt.savefig('./Output/predict_result.pdf', format="pdf", bbox_inches="tight")

system_name = platform.system()
if system_name == 'Windows':
    subprocess.run(['start', './Output/predict_result.pdf'], shell=True)
elif system_name == 'Darwin':
   subprocess.run(['open', './Output/predict_result.pdf'])
elif system_name == 'Linux':
    subprocess.run(['xdg-open', './Output/predict_result.pdf'])
else:
    print("The operating system is unknown")