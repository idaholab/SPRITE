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
from scipy.interpolate import interp1d
import time
import scipy.io as io
import math
import platform
import subprocess
import sys
sys.path.insert(0, './Tools')

from dataset_glass import DataSet
from net import DNN
import Physics

np.random.seed(1234)
tf.set_random_seed(1234)
tf.reset_default_graph()

x_dim = 1
x_num = 500
#input dimension for Branch Net
f_dim = x_num
#output dimension for Branch and Trunk Net
G_dim = x_num
#parameter dimension (screen,moi)
p_dim = 1

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

batch_size = 3
num_test = 5

data = DataSet(x_num, batch_size)

x_train, f_train, u_train, s_train, s_train_real = data.minibatch()
p_train = s_train

x_pos = tf.constant(x_train, dtype=tf.float32) #[x_num, x_dim]
x = tf.tile(x_pos[None, :, :], [batch_size, 1, 1]) #[batch_size, x_num, x_dim]
Xmin = 0
Xmax = 1

#placeholder for f
f_ph = tf.placeholder(shape=[None, f_dim], dtype=tf.float32)#[bs, f_dim]
u_ph = tf.placeholder(shape=[None, x_num], dtype=tf.float32)#[bs, x_num]
p_ph = tf.placeholder(shape=[None, p_dim], dtype=tf.float32)#[bs, x_num]

s_ph_real = tf.placeholder(shape=[None, x_dim], dtype=tf.float32)#[bs, x_num]

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

# loss
## data loss
N_sc = 200
loss1 = tf.reduce_mean(tf.square(u_ph[:,:N_sc] - u_pred[:,:N_sc]))
loss2 = tf.reduce_mean(tf.square(u_ph[:,N_sc+1:] - u_pred[:,N_sc+1:]))
weight = 0.999
loss_data = loss1*weight + loss2*(1-weight)

## physics loss
d_p_static = Physics.physic(x_num, batch_size, s_ph_real)
d_p_static_1d = tf.reshape(d_p_static,(batch_size,500**2))

layers_ex1 = [500**2] + [500]
W_ex1, b_ex1 = model.hyper_initial(layers_ex1)
d_p_static_f_1d_ex1 = model.fnn_B(d_p_static_1d, W_ex1, b_ex1)

d_p_static_f_1d_ex2 = d_p_static_f_1d_ex1*u_pp
layers_ex2 = [500] + [500**2]
W_ex2, b_ex2 = model.hyper_initial(layers_ex2)
d_p_static_f_1d = model.fnn_B(d_p_static_f_1d_ex2, W_ex2, b_ex2)

d_p_static_f = tf.reshape(d_p_static_f_1d,(batch_size,500,500))


f_ph_real = data.decode_f(f_ph)
fuser_model = tf.reverse(f_ph_real,axis=[1])
fuser_model_diff = fuser_model - tf.roll(fuser_model,(0, -1),axis=(0, 1))
temp_p = np.ones(x_num)
temp_p[-1] = 0
temp_p1 = np.diag(temp_p)
temp_p1 = tf.constant(temp_p1,dtype=tf.float32)
fuser_model_diff = tf.matmul(fuser_model_diff,temp_p1)
fuser_model_diff = tf.tile(fuser_model_diff[:,:,None], [1, 1, 1]) #[batch_size, _dim]

p_static = tf.matmul(d_p_static_f,fuser_model_diff)
p_static_cumulative = tf.cumsum(p_static, axis=1, exclusive=True, reverse=True)
p_static_cumulative = tf.reduce_sum(p_static_cumulative, axis=-1)
p_static_cumulative = tf.reverse(p_static_cumulative,axis=[1])

u_pred_real_temp = data.decode_u(u_pred)
u_pred_real = tf.reverse(u_pred_real_temp,axis=[1])
loss_physics1 = tf.reduce_mean(tf.square(u_pred_real[:,:N_sc] - p_static_cumulative[:,:N_sc]))
loss_physics2 = tf.reduce_mean(tf.square(u_pred_real[:,N_sc+1:] - p_static_cumulative[:,N_sc+1:]))
weight = 0.999
loss_physics = loss_physics1*weight + loss_physics2*(1-weight)

u_ph_real = data.decode_u(u_ph)
loss_physics_plus1 = tf.reduce_mean(tf.square(u_ph_real[:,:N_sc] - p_static_cumulative[:,:N_sc]))
loss_physics_plus2 = tf.reduce_mean(tf.square(u_ph_real[:,N_sc+1:] - p_static_cumulative[:,N_sc+1:]))
weight = 0.999
loss_physics_plus = loss_physics_plus1*weight + loss_physics_plus2*(1-weight)

loss_constraint = tf.reduce_mean(tf.maximum(u_pred_real-1, 0))

var_list = [f_cnn_f, b_cnn_f, W_g_f, b_g_f, W_g_x, b_g_x, f_cnn_p, b_cnn_p, W_p, b_p, W_ex1, b_ex1, W_ex2, b_ex2]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

load_model = tf.train.Saver([weight for weight in var_list])
load_model.restore(sess, '../../SPRITE-Data/largedata/PIDNO+/checkpoint_final_glass/model')

test_id, x_test, f_test, u_test, s_test, s_test_real = data.testbatch(num_test)
p_test = s_test
test_dict = {f_ph: f_test, u_ph: u_test, p_ph: p_test, s_ph_real: s_test_real}
u_x_test = model.fnn_T(x_test, W_g_x, b_g_x, Xmin, Xmax) #[num_test, x_num, G_dim]
u_pred1_test = u_f*u_x_test
u_pred2_test = u_p*u_x_test
u_pred_test = u_pred1_test-u_pred2_test

u_pred_test = tf.reduce_sum(u_pred_test, axis=-1)
u_pred_test_ = sess.run(u_pred_test, feed_dict=test_dict)
u_pred_test_ = data.decode_u(u_pred_test_)
f_test_real = data.decode_f(f_test)
s_test_real = data.decode_s(s_test)

err = np.mean(np.linalg.norm(u_test - u_pred_test_, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
#print('L2 error: %.3e'%err)
save_dict = {'test_id': test_id, 'x_test': x_test, 'f_test': f_test_real, 's_test': s_test_real, 'u_test': u_test, 'u_pred': u_pred_test_, 'l2': err}
io.savemat('./Output/Preds_TF_load_data0.04k.mat', save_dict)

tf.reset_default_graph()

while True:
    data_num = input("4 different milling duration cases are tested.\n\
You can choose these 4 cases, milling duration equations to \n\
21 min, 27 min, 30 min, 33 min, respectively, to see the prediction results.\n\
Enter the number of the test case you want to test (choose between 1~4):\nPlease enter here: ")
    
    try:
        data_num = int(data_num)
        if 1 <= data_num <= 4:
            if data_num == 1:
                data_num = 3-1
            elif data_num == 2:
                data_num = 4-1
            elif data_num == 3:
                data_num = 2-1
            elif data_num == 4:
                data_num = 5-1
            break
        else:
            print("############# Warning ############## \n\
Input out of range. Please enter an integer between 1 and 4.\n\
####################################")
    except ValueError:
        print("############# Warning ############## \n\
Invalid input. Please enter an integer.\n\
####################################")

x_test_plt = np.squeeze(x_test)
f_test_plt = np.squeeze(f_test_real[data_num,:])
p_plt = np.squeeze(u_test[data_num,:])
u_pred_plt = np.squeeze(u_pred_test_[data_num,:])
duration = np.squeeze(s_test_real[data_num])

cutsize = 250
fig, ax = plt.subplots()
ax.plot(x_test_plt, f_test_plt*100, 'k-', label="Feed")
ax.plot(x_test_plt[0:cutsize], p_plt[0:cutsize]*100, 'b-', label="Ref")
ax.plot(x_test_plt[0:cutsize], u_pred_plt[0:cutsize]*100, 'r--', label="PIDNO+")
ax.set_xlabel('Sieve size (mm)')
ax.set_ylabel('Cumulative product PSD (%)')
ax.set_title("duration = {} min; Model = PIDNO+".format(np.round(duration*60,2)))
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

