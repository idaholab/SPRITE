#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import tensorflow.compat.v1 as tf
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as io
import matplotlib.pyplot as plt

def physic(nbins, batch, screen):
    minY = 0.00001                      #minY (minimum broken particle size in the system) as a scalar
    nbins=nbins                           #nbins (Number of bins between max and min size for model) as a scalar
    Maxsize = 36.0          #Maxsize (Maximum sieve size of feed in mm) as a scalar
    Minsize = 0.1        #Minsize (Minimum sieve size of product in mm) as a scalar
    Carea=0.5
    Cfluid=2
    
    fmat, xWmmin, gamma, alpha, cdratio = [0.99981689453125, 1e-05, 0.1236419677734375, 0.3597564697265625, 0.5372467041015625]
    #fmat, xWmmin, gamma, alpha, cdratio = [0.417510986328125, 0.08025516921997071, 0.1447601318359375, 0.2417755126953125, 0.710418701171875]
    auser_model = np.linspace(round(Maxsize)+1, Minsize, nbins)
    # Wmmin
    x_IPS = auser_model     #initial partical size (unit:mm) big to small
    x_IPS = x_IPS/1000                                  #initial partical size (unit:m) 
    x_IPS[-1] = x_IPS[-1]+1e-5
    Wmmin = xWmmin/x_IPS
    # Wmkin
    v = 9.8                    
    Wmkin = v**2/2                                      #specific impact energy / specific kinetic energy
    
    k = 10                                              #maximum number of impacts
    
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
        X = np.expand_dims(X, axis=0)  # Add a new dimension at axis 0
        X = np.repeat(X, batch, axis=0) 
        X = tf.constant(X,dtype=tf.float32)
        # C                                             #Classification matrix
        dratio = x_IPS*1e3/screen
        Cratio = tf.exp(-((dratio)/0.02)*((1/dratio)**-2))*Carea*Cfluid
        Cratio = (Cratio-tf.math.reduce_min(Cratio))/(tf.math.reduce_max(Cratio)-tf.math.reduce_min(Cratio))
        C = tf.matrix_diag(Cratio)
        
        # m p r
        if ki == 1:
            d_m = X
            #m = m/np.sum(m)
            # p - equation (11)
            d_p_static = tf.matmul(C,d_m)
            # r - euqation (12)
            d_r = tf.matmul((I-C),d_m)
        else:
            d_m = tf.matmul(X,d_r)
            d_p_static = d_p_static + tf.matmul(C,d_m)
            d_r = tf.matmul((I-C),d_m)
            
    return d_p_static
