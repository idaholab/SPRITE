#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Minglei Lu
"""

import numpy as np
from scipy.interpolate import interp1d
from alive_progress import alive_bar
from Model_screen import myModel

def Enumeration(Input, Output, v, screen):
    ExpSieves = np.array(Output[['ExpSieves']])
    Expmass = np.array(Output[['Expmass']])
    
    def objective(x):
        auser_model, fuser_model, p_sieve_model, p_static_cumulative = myModel(Input, Output, v, x[0], x[1], x[2], x[3], x[4], screen=screen)
        f = interp1d(p_sieve_model, p_static_cumulative)
        p_transient_cumulative_point = f(ExpSieves)
        MSE = (np.square(np.subtract(p_transient_cumulative_point,Expmass))).mean()
        return MSE
    
    # initial guess
    fmat=0.891
    xWmmin=5.05E-02
    gamma=1.01E-04
    alpha=0.01683
    cdratio=6.09E-05
    
    resolution = 0.5
    fmatspace=[(1-resolution)*fmat,(1-0.5*resolution)*fmat,fmat,(1+(0.5*resolution))*fmat,(1+(resolution))*fmat]
    xWmminspace=[(1-resolution)*xWmmin,(1-0.5*resolution)*xWmmin,xWmmin,(1+(0.5*resolution))*xWmmin,(1+(resolution))*xWmmin]
    gammaspace=[(1-resolution)*gamma,(1-0.5*resolution)*gamma,gamma,(1+(0.5*resolution))*gamma,(1+(resolution))*gamma]
    alphaspace=[(1-resolution)*alpha,(1-0.5*resolution)*alpha,alpha,(1+(0.5*resolution))*alpha,(1+(resolution))*alpha]
    cdratiospace=[(1-resolution)*cdratio,(1-0.5*resolution)*cdratio,cdratio,(1+(0.5*resolution))*cdratio,(1+(resolution))*cdratio]
    
    x_all = np.zeros((5**5,5))
    MSE_all = np.zeros((5**5))
    i = 0
    with alive_bar(5**5, title='Processing', bar='filling') as bar: 
        for fmat in fmatspace:
            for xWmmin in xWmminspace:
                for gamma in gammaspace:
                    for alpha in alphaspace:
                        for cdratio in cdratiospace:
                            x = np.array([fmat,xWmmin,gamma,alpha,cdratio])
                            MSE = objective(x)
                            x_all[i] = x
                            MSE_all[i] = MSE
                            i = i+1
                            bar()
                        
    idx = np.argmin(MSE_all)
    x = x_all[idx]
    
    return x