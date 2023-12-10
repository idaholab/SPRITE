# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on Mon Jun  5 23:00:32 2023
@author: Minglei Lu

reference paper: 
    Miguel Gil a,⁎, Ennio Luciano b, Inmaculada Arauzo a. 
    Population balance model for biomass milling
    Powder Technology
"""

import numpy as np
from scipy.interpolate import interp1d

def myModel(Input, Output, v, fmat, xWmmin, gamma, alpha, cdratio, screen):
    # In[load data]
    FSieves = np.array(Input[['FSieves']])      #FSieves (Sieves used in feed PSD) as a vector. (PSD: partical size distribution)
    Feedmass = np.array(Input[['Feedmass']])    #Feedmass (Cumulative distribution of experimental feed %mass retained) as a matrix
    ExpSieves = np.array(Output[['ExpSieves']])  #ExpSieves (Sieves used in experimental PSD) as a vector
    
    minY = 0.00001                      #minY (minimum broken particle size in the system) as a scalar
    nbins=100                           #nbins (Number of bins between max and min size for model) as a scalar
    Maxsize = np.amax(FSieves)          #Maxsize (Maximum sieve size of feed in mm) as a scalar
    Minsize = np.amin(ExpSieves)        #Minsize (Minimum sieve size of product in mm) as a scalar
    Carea=0.5
    Cfluid=2
    
    # In[build parameters]
    # Fit
    x_temp = np.insert(FSieves, (0,len(FSieves)), (round(Maxsize)+1,0))
    y_temp = np.insert(Feedmass, (0,len(Feedmass)), (1,0))
    fit = interp1d(x_temp, y_temp, kind='linear',fill_value="extrapolate") 
    auser_model = np.linspace(round(Maxsize)+1, Minsize, nbins)
    fuser_model = fit(auser_model)
    
    # find feed diff
    fuser_model_diff = fuser_model - np.roll(fuser_model,-1)
    fuser_model_diff[-1] = fuser_model[-1]
    #stdd = np.std(fuser_model_diff)
    # Wmmin
    x_IPS = auser_model     #initial partical size (unit:mm) big to small
    x_IPS = x_IPS/1000                                  #initial partical size (unit:m) 
    x_IPS[-1] = x_IPS[-1]+1e-5
    Wmmin = xWmmin/x_IPS
    # Wmkin                    
    Wmkin = v**2/2                                      #specific impact energy / specific kinetic energy
    
    k = 10                                              #maximum number of impacts
    #p_transient_all = np.empty([nbins,k])
    r_all = np.empty([nbins,k])
    for ki in range(1,k+1):
        # q - equation (3)
        c = -gamma/((ki)*x_IPS*Wmkin)**alpha
        q = c*v + 1/cdratio*c                           #q=cv+d
        
        # B - equation (2)                                             #Breakage function
        B = np.identity(nbins)
        for i in range(1,nbins+1): # parents partical index
            for j in range(i+1,nbins+1): # child partical index
                B_temp = 0.5*(1+np.tanh((x_IPS[j-1]-minY)/minY))*(x_IPS[i-1]/x_IPS[j-1])**(q[i-1])
                B[j-1,i-1] = B_temp
        # b - equation (7)                                            #b_ij=Bj(i+1)-Bj(i)
        b = np.roll(B, 1, axis=0) - B
        b = np.tril(b,-1)
        
        # S - equation (1)                                            #Breakage probability
        S_temp = 1-np.exp(-fmat*k*x_IPS*(Wmkin-Wmmin))
        S = np.diag(S_temp)
        
        # X - equation (10)                                            #Breakage matrix
        I = np.identity(nbins)
        X = np.dot(b,S)+I-S
    
        # C                                             #Classification matrix
        screen = screen
        idx_fx = (np.abs(auser_model - screen)).argmin() #find index of size =< screen
        if auser_model[idx_fx] > screen:
            idx_fx = idx_fx-1
        else:
            idx_fx = idx_fx
        dratio = x_IPS[idx_fx:]*1e3/screen
        Cratio = np.exp(-((dratio)/0.02)*((1/dratio)**-2))*Carea*Cfluid
        Cratio = (Cratio-np.amin(Cratio))/(np.amax(Cratio)-np.amin(Cratio))
        C = np.zeros([nbins,nbins])
        C_temp = np.identity(nbins-idx_fx)                          
        C[idx_fx:,idx_fx:] = C_temp*Cratio
        # m p r
        if ki == 1:
            m = np.dot(X,fuser_model_diff)              #impacted material factor
            #m = m/np.sum(m)
            # p - equation (11)
            p_static = np.dot(C,m)
            # r - euqation (12)
            r = np.dot((I-C),m)
            r_all[:,ki-1] = r
        else:
            m = np.dot(X,r)
            p_static = p_static + np.dot(C,m)
            r = np.dot((I-C),m)
            r_all[:,ki-1] = r
            
    p_static_cumulative = p_static
    for index in range(1,nbins+1):
        p_static_temp = np.zeros(nbins)
        p_static_temp[:-index] = p_static[index:]
        p_static_cumulative = p_static_cumulative + p_static_temp
        
    p_sieve_model = np.insert(auser_model, (len(auser_model)), (0))
    auser_model = p_sieve_model
    fuser_model = np.insert(fuser_model, (len(fuser_model)), (0))
    p_static_cumulative = np.insert(p_static_cumulative, (len(p_static_cumulative)), (0))
   
    return auser_model, fuser_model, p_sieve_model, p_static_cumulative
