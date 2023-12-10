#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import platform
import subprocess
from Model_screen import myModel
from regression import regres

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
        v = input("Enter the milling speed in m/s: \nPlease enter here: ")
        try:
            v = float(v)
            if v > 0:
                print(f"\nMilling frequency = {v}m/s successfully input.\n")
                break
            else:
                print("\n############# Warning ############## \n\
Invalid input. Milling frequency should be a positive value.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input.\n\
####################################\n")
while True:
        moi = input("Enter the moisture content in %, for example trpe 40 as 40% of moisture: \nPlease enter here: ")
        try:
            moi = float(moi)
            if  0 <= moi <= 100:
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

Input = pd.read_csv("./Dataset/extra_input.csv")
Output = pd.read_csv("./Dataset/extra_output.csv")

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

if opt == 1:
    plt.plot(FSieves, Feedmass*100, '.', label="Feed PSD")
    plt.plot(ExpSieves, Expmass*100, 'b^', label="Expirenment")
    plt.plot(p_sieve_model, p_static_cumulative*100, 'r-', label="Model pred")
    plt.xlabel('Sieve size (mm)')
    plt.ylabel('Cumulative product PSD (%)')
    plt.legend()
elif opt == 0:
    plt.plot(FSieves, Feedmass*100, '.', label="Feed PSD")
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
