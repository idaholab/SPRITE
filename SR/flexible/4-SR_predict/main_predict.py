# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:57:35 2024

@author: minglel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import scipy.io as io
import sys
sys.path.insert(0, './Tools')
from scipy.interpolate import interp1d
import platform
import subprocess
from Model_screen import myModel
from Choose_DatasetNum import dataset
from get_equation import get_Equation

def reset_value(fitparameter):
    for i in range(len(fitparameter)):
        if fitparameter[i] > 1:
            fitparameter[i] = 1
        elif fitparameter[i] < 0:
            fitparameter[i] = 0
    return fitparameter

Complex_all = []
R2_all = []
for num_test in range(1,2):
    for data_num in range(1,2):
        start_time = time.perf_counter()
        for i in range(2,3):
            Input, Output, v, screen, moi = dataset(data_num)
            moi = moi/100
            
            FSieves = np.array(Input[['FSieves']])      #FSieves (Sieves used in feed PSD) as a vector. (PSD: partical size distribution)
            Feedmass = np.array(Input[['Feedmass']])    #Feedmass (Cumulative distribution of experimental feed %mass retained) as a matrix
            ExpSieves = np.array(Output[['ExpSieves']])  #ExpSieves (Sieves used in experimental PSD) as a vector
            Expmass = np.array(Output[['Expmass']])      #Expmass (Cumulative distribution of experimental product %mass retained) as a matrix
            
            '''
            moi = random.uniform(0.15, 1)
            moi = round(moi, 2)
            v = random.uniform(5, 10)
            v = round(v, 2)
            screen = random.uniform(5, 30)
            screen = round(screen, 2)
            '''
            
            #num_test = 1
            file_fmat = '../3-Equation_SR/Output/1-fmat/1-fmat_test{}.csv'.format(num_test)
            fmat_complex, fmat = get_Equation(file_fmat,moi,v,screen)
            file_xWmmin = '../3-Equation_SR/Output/2-xWmmin/2-xWmmin_test{}.csv'.format(num_test)
            xWmmin_complex, xWmmin = get_Equation(file_xWmmin,moi,v,screen)
            file_gamma = '../3-Equation_SR/Output/3-gamma/3-gamma_test{}.csv'.format(num_test)
            gamma_complex, gamma = get_Equation(file_gamma,moi,v,screen)
            file_alpha = '../3-Equation_SR/Output/4-alpha/4-alpha_test{}.csv'.format(num_test)
            alpha_complex, alpha = get_Equation(file_alpha,moi,v,screen)
            file_cdratio = '../3-Equation_SR/Output/5-cdratio/5-cdratio_test{}.csv'.format(num_test)
            cdratio_complex, cdratio = get_Equation(file_cdratio,moi,v,screen)
            
            fmat = reset_value(fmat)
            xWmmin = reset_value(xWmmin)
            gamma = reset_value(gamma)
            alpha = reset_value(alpha)
            cdratio = reset_value(cdratio)
            
            #i = 1
            x = [fmat[i], xWmmin[i], gamma[i], alpha[i], cdratio[i]]
            Complex = fmat_complex[i]+xWmmin_complex[i]+gamma_complex[i]+alpha_complex[i]+cdratio_complex[i]
            
            auser_model, fuser_model, p_sieve_model, p_static_cumulative = myModel(Input, Output, v, x[0], x[1], x[2], x[3], x[4], screen=screen)
            
            '''
            min_val = np.min(p_static_cumulative)
            max_val = np.max(p_static_cumulative)
            p_static_cumulative = (p_static_cumulative - min_val) / (max_val - min_val)
            '''
            
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
            #print('Coefficient of determination (R^2) = ', R2)
            R2_all.append(R2)
            Complex_all.append(Complex)
            
            #'''
            plt.plot(FSieves, Feedmass*100, '.', label="Feed PSD")
            plt.plot(ExpSieves, Expmass*100, 'b^', label="Expirenment")
            plt.plot(p_sieve_model, p_static_cumulative*100, 'r-', label="Model pred")
            plt.xlabel('Sieve size (mm)')
            plt.ylabel('Cumulative product PSD (%)')
            #plt.legend()
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
            #'''
            print('num_test{}/2, data_num{}/12, i:{}'.format(num_test,data_num,i))
        end_time = time.perf_counter()
        time_period = end_time-start_time
        print('use_time = ', time_period)
#save_dict = {'Complex_all': Complex_all, 'R2_all': R2_all}
#io.savemat('./Output/Complex_R2_single2.mat', save_dict)