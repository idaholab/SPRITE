#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import tensorflow.compat.v1 as tf
import numpy as np

class DNN:
    def __init__(self):
        pass

    #initialization for DNNs
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2.0/(in_dim+out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    #FNN
    def fnn_B(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i])) ## WA+b  tf.sigmoid tf.tanh
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])

        return Y

    def fnn_T(self, X, W, b, Xmin, Xmax):
        A = 2.*(X - Xmin)/(Xmax - Xmin) - 1.0
        A = tf.cast(A, dtype=tf.float32) ## Casts a tensor to a new type.
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))       #tf.sigmoid tf.tanh
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])

        return Y
    
    #--------------------additional function------------------
    def cnn_hyper_initial(self, layers):
        L = len(layers)
        b = []
        f = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2.0/(in_dim+out_dim))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            filters = tf.Variable(tf.random_normal(shape=[3, in_dim, out_dim], stddev=std))
            b.append(bias)
            f.append(filters)

        return f, b
    
    def cnn_B(self, X, f, b):
        A = X
        A = tf.expand_dims(A, axis=1)
        L = len(b)
        for i in range(L):
            conv1 = tf.nn.conv1d(A, f[i], stride=1, padding='SAME')
            activation1 = tf.nn.relu(tf.add(conv1, b[i]))
            A = tf.nn.pool(activation1, window_shape=[2], pooling_type='MAX', strides=[2], padding='SAME')
        flatten = tf.reshape(A, shape=[-1, A.shape[1] * A.shape[2]])
        
        return flatten
    
    def cnn_T(self, X, f, b, Xmin, Xmax):
        A = 2.*(X - Xmin)/(Xmax - Xmin) - 1.0
        A = tf.reshape(A, (1,tf.size(A)))
        A = tf.expand_dims(A, axis=1)
        A = tf.cast(A, dtype=tf.float32) ## Casts a tensor to a new type.
        L = len(b)
        for i in range(L):
            conv1 = tf.nn.conv1d(A, f[i], stride=1, padding='SAME')
            activation1 = tf.nn.relu(tf.add(conv1, b[i]))
            A = tf.nn.pool(activation1, window_shape=[2], pooling_type='MAX', strides=[2], padding='SAME')
        flatten = tf.reshape(A, shape=[-1, A.shape[1] * A.shape[2]])

        return flatten
    
