# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:15:42 2024

@author: minglel
"""

import numpy as np
import pandas as pd
from pysr import PySRRegressor

def SR_pysr(parameter):
    #df = pd.read_csv("../Data/regres_data_JRS-new.csv")
    df = pd.read_csv("../1-regression/regres_data_glasspowder.csv")
    INPUT = df[['duration']]
    OUTPUT = df[['fmat']]
    X = INPUT.values
    Y = OUTPUT.values
    y = Y[:,parameter]
    
    model = PySRRegressor(
        niterations=100,  # < Increase me for better results
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "exp",
            "square",
            #"sqrt",
            #"inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )
    
    model.fit(X, y)
    
    print(model)
        
        
        
        
    
    