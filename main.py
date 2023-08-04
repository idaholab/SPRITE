#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Minglei Lu
"""

import os

print('\n\
###################################################################### \n\
              ____    ____    ____    ___   _____   _____              \n\
             / ___|  |  _ \  |  _ \  |_ _| |_   _| | ____|             \n\
             \___ \  | |_) | | |_) |  | |    | |   |  _|               \n\
              ___) | |  __/  |  _ <   | |    | |   | |___              \n\
             |____/  |_|     |_| \_\ |___|   |_|   |_____|             \n\
                                                                       \n\
           Smart Preprocessing & Robust Integration Emulator           \n\
                                                                       \n\
                  \N{COPYRIGHT SIGN} 2023 Idaho National Laboratory    \n\
                                                                       \n\
###################################################################### \n\
\n\
SPRITE current functions include predicting \n\
the particle size distribution (PSD) of milled corn stover particles. \n\
')

while True:
        data_num = input("Choose the model for the prediction (choose between 1 and 3):\n\n\
[1] PBM: Population Balance Model \n\
[2] DNO+: Enhanced Deep Neural Operator \n\
[3] PIDNO+: Physics-informed DNO+ \n\
\nPlease enter here: ")
        if data_num == "":
            print('\nNo input detected, using Method No.1 as default.\n')
            data_num = 1
            os.chdir('./PBM')
            with open("./main.py") as f:
                exec(f.read())
            break
        try:
            data_num = int(data_num)
            if 1 <= data_num <= 3:
                print(f"\nMethod No.{data_num} is selected.\n")
                if data_num == 1:
                    os.chdir('./PBM')
                    with open("./main.py") as f:
                        exec(f.read())
                    os.chdir('../')
                elif data_num == 2:
                    os.chdir('./DNO+')
                    with open("./main.py") as f:
                        exec(f.read())
                    os.chdir('../')
                elif data_num == 3:
                    os.chdir('./PIDNO+')
                    with open("./main.py") as f:
                        exec(f.read())
                    os.chdir('../')
                break
            else:
                print("\n############# Warning ############## \n\
Input out of range. Please enter an integer between 1 and 3.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input. Please enter an integer.\n\
####################################\n")

else:
    print('Invalid input')
