# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:13:25 2024

@author: minglel
"""

import os
import shutil
import sys
sys.path.insert(0, './Tools')
from def_SR import SR_pysr

for parameter in range(0,5):
    for idx in range(1,5):
        if parameter == 0:
            name = '1-fmat'
        elif parameter == 1:
            name = '2-xWmmin'
        elif parameter == 2:
            name = '3-gamma'
        elif parameter == 3:
            name = '4-alpha'
        elif parameter == 4:
            name = '5-cdratio'
        
        SR_pysr(parameter)
        
        # Current directory
        current_directory = os.getcwd()
        # Directory to move the file to
        new_directory = os.path.join(current_directory, './Output_new/{}').format(name)
        # Ensure the new directory exists, create it if it doesn't
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        # Iterate over files in the current directory
        for filename in os.listdir(current_directory):
            if filename.endswith('.bkup') or filename.endswith('.pkl'):
                # Construct the full path to the file
                file_path = os.path.join(current_directory, filename)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted '{filename}'")
            if filename.endswith('.csv'):
                # Change filename
                new_filename = '{}_test{}.csv'.format(name,idx)
                # Move the file to the new directory with the new filename
                shutil.move(os.path.join(current_directory, filename), os.path.join(new_directory, new_filename))
                print(f"Moved '{filename}' to '{os.path.join(new_directory, new_filename)}'")
                
                