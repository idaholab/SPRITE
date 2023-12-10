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
import time
import webbrowser
import shutil
import stat
import scipy.io as io
import platform
import subprocess
import sys
sys.path.insert(0, './Tools')

print('#################################################### \n \
This is the DNO+ model used to predict the milled corn stover particle size distribution (PSD). \n \
100 testing data is prepared. \n\
#################################################### \n\
You can select one of them to see the prediction results after training. \n')
    
method = input("You can choose to train the DNO+ right now and give prediction after training, \n\
or choose to load the pre-trained model and give prediction right now:\n\
[1] Load pre-trained model \n\
[2] Train the DNO+ model \n\
[3] Continue training \n\
[4] (Re)Download pre-trained model \n\
[5] Predict new data \n\
Please enter here: " )

if method == '2':
    import main_DNOplus
    raise SystemExit()
elif method == '5':
    import main_DNOplus_prednew
    raise SystemExit()
elif method == '3':
    print(os.getcwd())
    if not os.path.exists('./Tools/checkpoint/'):
        print("\nCannot continue training, as the pre-trained model does not exist. Please run method [1] first and then run method [2].\n")
        raise SystemExit()
    import main_DNOplus_continue
    raise SystemExit()
elif method == '4':
    directory_path = os.path.join('../../', 'SPRITE-Data')
    if os.path.exists(directory_path):
        way = input('Directory exist already, choose to abort or overwrite the original SPRITE-Data folder.\n\
[1] Overwrite (Cautions! this option will overwrite the original folder.\n\
[2] Abort\n\
Please enter here: ')
        if way == '1':
            def make_dir_writable(function, path, exception):
                """The path on Windows cannot be gracefully removed due to being read-only,
                so we make the directory writable on a failure and retry the original function.
                """
                os.chmod(path, stat.S_IWRITE)
                function(path)
            shutil.rmtree(directory_path, onerror=make_dir_writable)
            print("\nDirectory removed successfully.\n")
            
            os.makedirs(directory_path)
            clone_command = "git lfs install\n\git clone https://github.com/yidongxiainl/SPRITE-Data.git" 
            clone_with_path = clone_command  +" "+ directory_path
            os.system(clone_with_path)
            print("\nData download successfully.\n")
            raise SystemExit()
        else:
            print('Code abort.')
    if not os.path.exists('../../SPRITE-Data'):
        print("\nThe pre-trained model does not exist, the file 'SPRITE-Data' does not exist. Please download the pre-trained model first.\n")
        
        option = input("You can choose to download the pre-trained model now or choose to exit:\n\
[1] Open download link, and download by yourself. \n\
[2] Auto download. \n\
[3] Exit the code.\n\
Please enter here: " )
        if option == "1":
            webbrowser.open('https://github.com/yidongxiainl/SPRITE-Data.git')
            print('\nDownload the checkpoint file in the directory at the same level as OpenPBM and restart the code main.py\n')
            raise SystemExit()
        elif option == '2':
            directory_path = os.path.join('../../', 'SPRITE-Data')
            os.makedirs(directory_path)
            clone_command = "git lfs install\n\git clone https://github.com/yidongxiainl/SPRITE-Data.git" 
            clone_with_path = clone_command  +" "+ directory_path
            os.system(clone_with_path)
            print("\nData download successfully. Please rerun the code.\n")
            raise SystemExit()
        else:
            if option == '3':
                raise SystemExit()
            else:
                print('\nInvalid input, exit as default.\n')
                raise SystemExit()

else:
    if method == '1':
        print('\nMethod 1 is selected\n')
    else:
        print('\nInvalid input, using method 1 as default.\n')
    
    if not os.path.exists('../../SPRITE-Data'):
        print("\nThe pre-trained model does not exist, the file 'SPRITE-Data' does not exist. Please download the pre-trained model first.\n")
        raise SystemExit()

    while True:
            data_num = input("Enter the number of the data you want to test (choose between 1~100):\nPlease enter here: ")
            if data_num == "":
                print('\nNo input detected, using 1st dataset as default.\n')
                data_num = 1
                break
            try:
                data_num = int(data_num)
                if 1 <= data_num <= 100:
                    print(f"\nData No.{data_num} is selected.\n")
                    # Your main logic goes here
                    break
                else:
                    print("\n############# Warning ############## \n\
    Input out of range. Please enter an integer between 1 and 100.\n\
    ####################################\n")
            except ValueError:
                print("\n############# Warning ############## \n\
    Invalid input. Please enter an integer.\n\
    ####################################\n")
    
    from Load_DNOplus import modelload
    tf.reset_default_graph()
    start_time1 = time.perf_counter()
    #load data
    df1 = io.loadmat('./Dataset/JRS_60Hz/input_{}.mat'.format(data_num))
    FSieves = df1.get('FSieves')
    fs = np.flip(FSieves)
    Feedmass = df1.get('Feedmass')
    fm = np.flip(Feedmass)
    screen = df1.get('screen')
    moi = df1.get('moi')
    Pmass = df1.get('Pmass')
    p = np.flip(Pmass)
    #pred
    start_time2 = time.perf_counter()
    x_test, f_test, s_test_real, m_test_real, u_pred_test_ = modelload(fs,fm,screen,moi)
    
    end_time = time.perf_counter()
    print('Run time: %.3f seconds'%(end_time - start_time1),'\nPred time: %.3f seconds'%(end_time - start_time2))
    #save
    save_dict = {'x_test': x_test, 'f_test': f_test, 'u_test':p, 's_test': s_test_real, 'm_test': m_test_real, 'u_pred': u_pred_test_}
    io.savemat('./Output/Preds_TF_loadsingle_data{}.mat'.format(data_num), save_dict)
    
    x_test_plt = np.squeeze(x_test)*35
    p_plt = np.squeeze(p)
    u_pred_plt = np.squeeze(u_pred_test_)
    fm_plt = np.squeeze(fm)
    moi_plt = np.squeeze(moi)
    screen_plt = np.squeeze(screen)
    
    fig, ax = plt.subplots()
    ax.plot(x_test_plt, fm_plt*100, 'k-', label="Feed")
    ax.plot(x_test_plt[0:200], p_plt[0:200]*100, 'b-', label="Exp")
    ax.plot(x_test_plt[0:200], u_pred_plt[0:200]*100, 'r--', label="DNO+")
    ax.set_xlabel('Sieve size (mm)')
    ax.set_ylabel('Cumulative product PSD (%)')
    ax.set_title("M.C. = {} %; Screen size = {} mm; Model = DNO+".format(np.round(moi_plt*100,2),np.round(screen_plt,2)))
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
