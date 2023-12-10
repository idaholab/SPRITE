#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import platform
import subprocess
from scipy.interpolate import interp1d
from Load_DNOplus import modelload

while True:
        option = input("Do you have the feed PSD data ready? \nPlease enter here ( [y]/n ): ")
        try:
            if option == 'y':
                print("Feed PSD data is loaded.\n")
                break
            elif option == '':
                print("Feed PSD data is loaded as default.\n")
                break
            elif option == 'n':
                print("Please prepare the feed PSD data as input first.\n")
                raise SystemExit()
            else:
                print("\n############# Warning ############## \n\
Invalid input. Please enter ( y/n ).\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input. Please enter ( y/n ).\n\
####################################\n")
while True:
        option = input("Do you have the product PSD data ready used to validate the predition results? \nPlease enter here ( [y]/n ): ")
        try:
            if option == 'y':
                opt = 1
                print("Product PSD data is loaded.\n")
                break
            elif option == '':
                opt = 1
                print("Product PSD data is loaded as default.\n")
                break
            elif option == 'n':
                opt = 0
                print("No product PSD data.\n")
                break
            else:
                print("\n############# Warning ############## \n\
Invalid input. Please enter ( y/n ).\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input. Please enter ( y/n ).\n\
####################################\n")
while True:
        moi = input("Enter the moisture content in %, for example trpe 40 as 40% of moisture: \nPlease enter here: ")
        try:
            moi = float(moi)
            if  0 <= moi <= 100:
                moi = moi/100
                print(f"\nMoisture = {moi}% successfully input.\n")
                break
            else:
                print("\n############# Warning ############## \n\
Invalid input. Moisture should be in the range between 0~100%.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input.\n\
####################################\n")
while True:
        screen = input("Enter the screen size in mm: \nPlease enter here: ")
        try:
            screen = float(screen)
            if  screen > 0:
                print(f"\nMoisture = {moi}% successfully input.\n")
                break
            else:
                print("\n############# Warning ############## \n\
Invalid input. Moisture should be in the range between 0~100%.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input.\n\
####################################\n")
tf.reset_default_graph()
start_time1 = time.perf_counter()
#load data
Input = pd.read_csv("../PBM/Dataset/extra_input.csv")
Output = pd.read_csv("../PBM/Dataset/extra_output.csv")

FSieves = np.array(Input[['FSieves']])      #FSieves (Sieves used in feed PSD) as a vector. (PSD: partical size distribution)
Feedmass = np.array(Input[['Feedmass']])    #Feedmass (Cumulative distribution of experimental feed %mass retained) as a matrix
ExpSieves = np.array(Output[['ExpSieves']])  #ExpSieves (Sieves used in experimental PSD) as a vector
Expmass = np.array(Output[['Expmass']])      #Expmass (Cumulative distribution of experimental product %mass retained) as a matrix

nbins=500
x_temp = np.insert(FSieves, (0,len(FSieves)), (35))
y_temp = np.insert(Feedmass, (0,len(Feedmass)), (1,0))
fit = interp1d(x_temp, y_temp, kind='linear',fill_value="extrapolate") 
auser_model = np.linspace(35, 0, nbins)
fuser_model = fit(auser_model)
fs = np.flip(auser_model)
fm = np.flip(fuser_model)

start_time2 = time.perf_counter()
x_test, f_test, s_test_real, m_test_real, u_pred_test_ = modelload(fs,fm,screen,moi)

end_time = time.perf_counter()
print('Run time: %.3f seconds'%(end_time - start_time1),'\nPred time: %.3f seconds'%(end_time - start_time2))

x_test_plt = np.squeeze(x_test)*35
u_pred_plt = np.squeeze(u_pred_test_)
moi_plt = np.squeeze(moi)
screen_plt = np.squeeze(screen)
fm_plt = np.squeeze(fm)

if opt == 0:    
    fig, ax = plt.subplots()
    ax.plot(x_test_plt, fm_plt*100, 'k-', label="Feed")
    ax.plot(x_test_plt[0:200], u_pred_plt[0:200]*100, 'r--', label="DNO+")
    ax.set_xlabel('Sieve size (mm)')
    ax.set_ylabel('Cumulative product PSD (%)')
    ax.set_title("M.C. = {} %; Screen size = {} mm; Model = DNO+".format(np.round(moi_plt*100,2),np.round(screen_plt,2)))
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.ylim([-5,105])
    plt.savefig('./Output/predict_result.pdf', format="pdf", bbox_inches="tight")
elif opt == 1:
    fig, ax = plt.subplots()
    ax.plot(x_test_plt, fm_plt*100, 'k-', label="Feed")
    ax.plot(ExpSieves, Expmass*100, 'b^', label="Exp")
    ax.plot(x_test_plt[0:200], u_pred_plt[0:200]*100, 'r--', label="DNO+")
    ax.set_xlabel('Sieve size (mm)')
    ax.set_ylabel('Cumulative product PSD (%)')
    ax.set_title("M.C. = {} %; Screen size = {} mm; Model = DNO+".format(np.round(moi_plt*100,2),np.round(screen_plt,2)))
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.ylim([-5,105])
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
print('\nProcess completed successfully.\n')
