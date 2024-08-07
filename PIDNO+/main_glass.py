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
import webbrowser
import shutil
import stat
import sys
sys.path.insert(0, './Tools')

print('#################################################### \n \
This is the PIDNO+ model used to predict the milled corn stover particle size distribution (PSD). \n \
100 testing data is prepared. \n\
#################################################### \n\
You can select one of them to see the prediction results. \n\
    ')

method = input("You can choose to train the PIDNO+ right now and give prediction after training, \n\
or choose to load the pre-trained model and give prediction right now:\n\
[1] Load pre-trained model \n\
[2] Train the PIDNO+ model \n\
[3] Continue training model \n\
[4] (Re)Download pre-trained model \n\
Please enter here: " )

if method == '2':
    import main_PIDNOplus_glass
elif method == '3':
    if not os.path.exists('./Tools/checkpoint'):
        print("\nCannot continue training, as the pre-trained model does not exist. Please run method [1] first and then run method [2].\n")
        raise SystemExit()
    import main_PIDNOplus_glass_continue
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
        print("\nThe file '/SPRITE-Data' does not exist. Please download the pre-trained model first.\n")
        raise SystemExit()
        
    import Load_PIDNOplus_glass
print('\nProcess completed successfully.\n')
