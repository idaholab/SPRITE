#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Minglei Lu
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from scipy.interpolate import interp1d
from net import DNN

def modelload(fs,fm, screen, moi):
    tf.reset_default_graph()
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    fs = np.squeeze(fs)
    fm = np.squeeze(fm)
    
    f = interp1d(fs,fm)
    auser_model = np.linspace(35,0,500)
    fuser_model = f(auser_model)
    
    # data info
    x_train = np.load('./Dataset/x_train.npy')
    f_train_mean = np.load('./Dataset/f_train_mean.npy')
    f_train_std = np.load('./Dataset/f_train_std.npy')
    u_train_mean = np.load('./Dataset/u_train_mean.npy')
    u_train_std = np.load('./Dataset/u_train_std.npy')
    m_train_mean = np.array([[0.49989044]])
    m_train_std = np.array([[0.28840904]])
    s_train_mean = np.array([[20.01425878]])
    s_train_std = np.array([[5.7707564]])
    
    #layers
    x_dim = 1
    x_num = 500
    #input dimension for Branch Net
    f_dim = x_num
    #output dimension for Branch and Trunk Net
    G_dim = x_num
    #parameter dimension (screen,moi)
    p_dim = 2
    
    #CNN Net - feed PSD
    layers_c_f = [f_dim] + [200]*3 + [f_dim]
    #CNN Net - parameters
    layers_c_p = [p_dim] + [200]*3 + [G_dim]
    #Branch Net 300-2_200-5_100-2
    layers_f = [f_dim] + [200]*3 + [G_dim]
    #Trunk Net
    layers_x = [x_dim] + [200]*3 + [G_dim]
    #fnn net (after cnn net)
    layers_p = [G_dim] + [200]*3 + [G_dim]
    
    batch_size = 1
    
    x_pos = tf.constant(x_train, dtype=tf.float32) #[x_num, x_dim]
    x = tf.tile(x_pos[None, :, :], [batch_size, 1, 1]) #[batch_size, x_num, x_dim]
    Xmin = 0
    Xmax = 1
    
    #placeholder for f
    f_ph = tf.placeholder(shape=[None, f_dim], dtype=tf.float32)#[bs, f_dim]
    p_ph = tf.placeholder(shape=[None, p_dim], dtype=tf.float32)#[bs, x_num]
    
    #model
    model = DNN()
    
    f_cnn_f, b_cnn_f = model.cnn_hyper_initial(layers_c_f)
    W_g_f, b_g_f = model.hyper_initial(layers_f)
    
    W_g_x, b_g_x = model.hyper_initial(layers_x)
    
    f_cnn_p, b_cnn_p = model.cnn_hyper_initial(layers_c_p)
    W_p, b_p = model.hyper_initial(layers_p)
    
    u_ff = model.cnn_B(f_ph, f_cnn_f, b_cnn_f)
    u_f = model.fnn_B(u_ff, W_g_f, b_g_f)
    u_f = tf.tile(u_f[:, None, :], [1, x_num, 1]) #[batch_size, x_num, G_dim]
    
    u_x = model.fnn_T(x, W_g_x, b_g_x, Xmin, Xmax) #[batch_size, x_num, G_dim]
    u_pred1 = u_f*u_x
    
    u_pp = model.cnn_B(p_ph, f_cnn_p, b_cnn_p)
    u_p = model.fnn_B(u_pp, W_p, b_p)
    u_p = tf.tile(u_p[:, None, :], [1, x_num, 1]) #[batch_size, x_num, G_dim]
    u_pred2 = u_p*u_x
    
    u_pred = u_pred1-u_pred2
    u_pred = tf.reduce_sum(u_pred, axis=-1)
    
    var_list = [f_cnn_f, b_cnn_f, W_g_f, b_g_f, W_g_x, b_g_x, f_cnn_p, b_cnn_p, W_p, b_p]
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #load model
    load_model = tf.train.Saver([weight for weight in var_list])
    load_model.restore(sess, '../../SPRITE-Data/largedata/DNO+/checkpoint_data0.3k_final/model')
    #load_model.restore(sess, './Tools/checkpoint/model')
    
    x_test = x_train
    f_test = fuser_model
    s_test = screen
    m_test = moi
    
    F_test = (f_test - f_train_mean)/(f_train_std + 1.0e-6)
    S_test = (s_test - s_train_mean)/(s_train_std + 1.0e-6)
    M_test = (m_test - m_train_mean)/(m_train_std + 1.0e-6)
    
    P_test = np.concatenate((S_test,M_test),axis=1)
    
    test_dict = {f_ph: F_test, p_ph: P_test}
    u_x_test = model.fnn_T(x_test, W_g_x, b_g_x, Xmin, Xmax) #[num_test, x_num, G_dim]
    u_pred1_test = u_f*u_x_test
    u_pred2_test = u_p*u_x_test
    u_pred_test = u_pred1_test-u_pred2_test
    
    u_pred_test = tf.reduce_sum(u_pred_test, axis=-1)
    u_pred_test_ = sess.run(u_pred_test, feed_dict=test_dict)
    u_pred_test_ = u_pred_test_*(u_train_std + 1.0e-6) + u_train_mean
    
    s_test_real = S_test*(s_train_std + 1.0e-6) + s_train_mean
    m_test_real = M_test*(m_train_std + 1.0e-6) + m_train_mean
    
    # reset variable name number
    tf.reset_default_graph()
    return x_test, f_test, s_test_real, m_test_real, u_pred_test_
