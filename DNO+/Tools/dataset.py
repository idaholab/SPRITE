#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import pandas as pd
np.random.seed(1234)

class DataSet:
    def __init__(self, N, batch_size):
        self.N = N
        self.batch_size = batch_size
        self.x_train, self.F_train, self.U_train, self.S_train, self.M_train,\
            self.F_test, self.U_test, self.S_test, self.M_test,\
            self.u_train_mean, self.u_train_std, self.s_train_mean, self.s_train_std, \
                self.m_train_mean, self.m_train_std = self.samples()
        
    def decode(self, x):
        return x*(self.u_train_std + 1.0e-6) + self.u_train_mean
    
    def decode_s(self, x):
        return x*(self.s_train_std + 1.0e-6) + self.s_train_mean
    
    def decode_m(self, x):
        return x*(self.m_train_std + 1.0e-6) + self.m_train_mean
    
    def samples(self):
        Depth = 400
        num_test = 100
        num_train = Depth - num_test
        
        # ----------------input---------------------------
        '''
        df = io.loadmat('./Dataset/JRS_60Hz/input_1.mat')
        FSieves = df.get('FSieves')
        Data_input = np.zeros((Depth,int(np.shape(FSieves)[1])))
        Data_output = np.zeros((Depth,int(np.shape(FSieves)[1])))
        screen_all = np.zeros((Depth,1))
        moi_all = np.zeros((Depth,1))
        for i in range(0,Depth):
            df1 = io.loadmat('./Dataset/JRS_60Hz/input_{}.mat'.format(i+1))
            Feedmass = df1.get('Feedmass')
            f = np.flip(Feedmass)
            Data_input[i,:] = f
            Pmass = df1.get('Pmass')
            p = np.flip(Pmass)
            Data_output[i,:] = p
            screen = df1.get('screen')
            screen_all[i,:] = screen
            moi = df1.get('moi')
            moi_all[i,:] = moi
            
            if i%1000 == 0:
                print('Loading data ... {:>5}%.'.format(np.round(i/Depth,decimals=2)*100))
        print('Complete loading.')
        '''
        
        Data_input = np.load('./Dataset/JRS_60Hz/Data_input.npy')
        Data_output = np.load('./Dataset/JRS_60Hz/Data_output.npy')
        screen_all = np.load('./Dataset/JRS_60Hz/screen_all.npy')
        moi_all = np.load('./Dataset/JRS_60Hz/moi_all.npy')
        FSieves = np.load('./Dataset/JRS_60Hz/FSieves.npy')
        
        inputdata = Data_input
        outputdata = Data_output

        x_train = np.flip(FSieves)
        x_train = (x_train-np.amin(x_train))/(np.amax(x_train)-np.amin(x_train))
        x_train = np.reshape(x_train,(int(np.shape(x_train)[1]),1))
        F = inputdata
        U = outputdata
        
        # ----------------process---------------------------
        F_train = F[:num_train, :]
        U_train = U[:num_train, :]
        F_test = F[-num_test:, :]
        U_test = U[-num_test:, :]
        S_train = screen_all[:num_train, :]
        S_test = screen_all[-num_test:, :]
        M_train = moi_all[:num_train, :]
        M_test = moi_all[-num_test:, :]
        
        f_train_mean = np.mean(F_train, axis=0, keepdims=True)
        f_train_std = np.std(F_train, axis=0, keepdims=True)
        u_train_mean = np.mean(U_train, axis=0, keepdims=True)
        u_train_std = np.std(U_train, axis=0, keepdims=True)
        s_train_mean = np.mean(S_train, axis=0, keepdims=True)
        s_train_std = np.std(S_train, axis=0, keepdims=True)
        m_train_mean = np.mean(M_train, axis=0, keepdims=True)
        m_train_std = np.std(M_train, axis=0, keepdims=True)
        
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-6)
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-6)
        S_train = (S_train - s_train_mean)/(s_train_std + 1.0e-6)
        M_train = (M_train - m_train_mean)/(m_train_std + 1.0e-6)

        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-6)
        S_test = (S_test - s_train_mean)/(s_train_std + 1.0e-6)
        M_test = (M_test - m_train_mean)/(m_train_std + 1.0e-6)
        
        return x_train, F_train, U_train, S_train, M_train, F_test, U_test, S_test, M_test, \
            u_train_mean, u_train_std, s_train_mean, s_train_std, m_train_mean, m_train_std
               
    def minibatch(self):    ## random select one training data from f & u
        batch_id = np.random.choice(self.F_train.shape[0], self.batch_size, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)
        s_train = [self.S_train[i:i+1] for i in batch_id]
        s_train = np.concatenate(s_train, axis=0)
        m_train = [self.M_train[i:i+1] for i in batch_id]
        m_train = np.concatenate(m_train, axis=0)

        return self.x_train, f_train, u_train, s_train, m_train

    def testbatch(self, num_test):
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)
        s_test = [self.S_test[i:i+1] for i in batch_id]
        s_test = np.concatenate(s_test, axis=0)
        m_test = [self.M_test[i:i+1] for i in batch_id]
        m_test = np.concatenate(m_test, axis=0)
        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, self.x_train, f_test, u_test, s_test, m_test
