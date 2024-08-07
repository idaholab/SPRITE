# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:16:31 2024

@author: minglel
"""

import os

print('#################################################### \n \
This is the SR model used to find the explicit \n \
expression of the comminution process. \n\
#################################################### \n\
    ')

while True:
        function = input("Choose the function of SR (choose between 1 and 2):\n\n\
[1] Find explicit expression\n\
[2] Test explicit expression\n\
\nPlease enter here: ")
        if function == "":
            print('\nNo input detected, using function No.1 as default.\n')
            os.chdir('./3-Equation_SR')
            with open("./main_SR.py") as f:
                exec(f.read())
            break
        try:
            function = int(function)
            if 1 <= function <= 2:
                print(f"\nfunction No.{function} is selected.\n")
                if function == 1:
                    os.chdir('./3-Equation_SR')
                    with open("./main_SR.py") as f:
                        exec(f.read())
                    os.chdir('../')
                elif function == 2:
                    os.chdir('./4-SR_predict')
                    with open("./main_predict.py") as f:
                        exec(f.read())
                    os.chdir('../')
                break
            else:
                print("\n############# Warning ############## \n\
Input out of range. Please enter an integer between 1 and 2.\n\
####################################\n")
        except ValueError:
            print("\n############# Warning ############## \n\
Invalid input. Please enter an integer.\n\
####################################\n")