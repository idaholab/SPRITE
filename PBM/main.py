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
import time
import platform
import subprocess
import scipy.io as io
import os
import sys
sys.path.insert(0, './Tools')
from Model_screen import myModel
from Enumeration_solver import Enumeration
from GA_solver import GA_self


def print_table(headers, data):
    # Determine the width for each column
    column_widths = [max(len(str(header)), max(len(str(row[i])) for row in data)) for i, header in enumerate(headers)]

    # Print the table headers
    for i, header in enumerate(headers):
        print(f"{header:<{column_widths[i]}}", end=" | ")  # Left-aligned header with padding

    print()  # Newline after printing headers

    # Print the table data
    for row in data:
        for i, cell in enumerate(row):
            print(f"{cell:<{column_widths[i]}}", end=" | ")  # Left-aligned data with padding

        print()  # Newline after printing a row

print('#################################################### \n\
This is the Population balance model (PBM) used to predict the milled corn stover particle size distribution (PSD). \n \
The data is prepared as examples to show the function of the model. \n\
#################################################### ')

mode = input("Enter the mode you want choose \n\
[1] Data fitting mode \n\
[2] Model prediction mode \n\
Please enter here: ")

table_headers = ["No.","Mill","Freq.(Hz)","Moi.(%)","screen(mm)"]
table_data = [
    [1,"JRS",40,20,12.70],
    [2,"JRS",40,20,19.05],
    [3,"JRS",40,40,12.70],
    [4,"JRS",40,40,19.05],
    [5,"JRS",50,20,12.70],
    [6,"JRS",50,20,19.05],
    [7,"JRS",50,40,12.70],
    [8,"JRS",50,40,19.05],
    [9,"JRS",60,20,12.70],
    [10,"JRS",60,20,19.05],
    [11,"JRS",60,40,12.70],
    [12,"JRS",60,40,19.05],
    [13,"Wiley",40,'dry',10],
    [14,"Wiley",60,'dry',10],
    [15,"Wiley",60,20,10],
    [16,"Wiley",60,40,10],
]

if mode == '1':
    print_table(table_headers, table_data)
    while True:
            data_num = input("Enter the No. data you want choose (choose between 1~10)\n\
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
    def main():
        # load data
        start_time = time.perf_counter()
        from Choose_DatasetNum import dataset
        Input, Output, v, screen, moi = dataset(data_num)
        
        method = input("Enter the optimization you want choose \n [1] Genetic algorithm \n [2] Enumeration \nPlease enter here: ")
        if method == '2':
            opt = Enumeration
        elif method == '1':
            opt = GA_self
        else:
            print('\nInvalid input, using GA method as default.\n')
            opt = GA_self
            
        FSieves = np.array(Input[['FSieves']])      #FSieves (Sieves used in feed PSD) as a vector. (PSD: partical size distribution)
        Feedmass = np.array(Input[['Feedmass']])    #Feedmass (Cumulative distribution of experimental feed %mass retained) as a matrix
        ExpSieves = np.array(Output[['ExpSieves']])  #ExpSieves (Sieves used in experimental PSD) as a vector
        Expmass = np.array(Output[['Expmass']])      #Expmass (Cumulative distribution of experimental product %mass retained) as a matrix
        
        x = opt(Input, Output, v, screen)
        auser_model, fuser_model, p_sieve_model, p_static_cumulative = myModel(Input, Output, v, x[0], x[1], x[2], x[3], x[4], screen=screen)
        end_time = time.perf_counter() 
        print('Training time:', end_time-start_time)
        # R^2
        f = interp1d(p_sieve_model, p_static_cumulative)
        p_transient_cumulative_point = f(ExpSieves)
        Expmass_mean = np.mean(Expmass)
        SST = np.sum(np.square(Expmass-Expmass_mean))
        SSR = np.sum(np.square(p_transient_cumulative_point-Expmass))
        R2 = 1-SSR/SST
        print('Coefficient of determination (R^2) = ', R2)
        
        # save data
        df = pd.DataFrame({'sieve': p_sieve_model,'cumulative PSD': p_static_cumulative})
        df.to_csv('./Output/PBM_data{}.csv'.format(data_num), index=False)
        
        save_dict = {'FSieves': FSieves, 'Feedmass': Feedmass, 'ExpSieves':ExpSieves, 'Expmass': Expmass, 'p_sieve_model': p_sieve_model, 'p_static_cumulative': p_static_cumulative, 'moi': moi, 'screen': screen}
        io.savemat('./Output/PBM_data{}.mat'.format(data_num), save_dict)
        
        fig, ax = plt.subplots()
        #ax.plot(FSieves, Feedmass, '.', auser_model, fuser_model*100, '-', label="Feed")
        ax.plot(FSieves, Feedmass*100, '.', label="Feed")
        ax.plot(ExpSieves, Expmass*100, 'b^', label="Exp")
        ax.plot(p_sieve_model, p_static_cumulative*100, 'r-', label="Model")
        ax.set_xlabel('Sieve size (mm)')
        ax.set_ylabel('Cumulative product PSD (%)')
        ax.set_title("M.C. = {} %; Screen size = {} mm; Model = PBM".format(moi,np.round(screen,2)))
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
    if __name__ == '__main__':
        main()
elif mode == '2':
    print_table(table_headers, table_data)
    option = input("Enter the option you want choose \n\
[1] Predict existing data to evaluate prediction accuracy \n\
[2] Predict new data (You can prepare the data by yourself) \n\
\nPlease enter here: ")
    if option == '1':
        import PBM_pred_exist
        raise SystemExit()
    elif option == '2':
        import PBM_pred_extra
        raise SystemExit()
    else:
        print("Invalid input.\n")
else:
    print("Invalid input.\n")
