# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:55:32 2024

@author: minglel
"""

import pandas as pd
import sympy as sp
import re

def get_Equation(filepath,duration):
    #filepath = '../output/3-gamma/3-gamma_test1.xlsx'
    df = pd.read_csv(filepath)
    df['Equation'] = df['Equation'].astype(str)
    
    def create_function(equation):
        # Replace custom variable names with symbols
        #equation = re.sub(r'cube\(([^)]+)\)', r'(\1)**3', equation)
        equation = equation.replace('neg', '-').replace('inv', '1/').replace('cube', '')
        
        # Define the symbols
        x0 = sp.symbols('x0')
        
        # Parse the equation string into a sympy expression
        expr = sp.sympify(equation)
        
        # Convert the sympy expression into a Python function
        func = sp.lambdify((x0), expr, 'numpy')
        return func
    
    df['function'] = df['Equation'].apply(create_function)
    
    # Example inputs
    '''
    moi = 1.0
    v = 2.0
    sc = 3
    '''
    
    # Evaluate the functions
    results = df['function'].apply(lambda f: f(duration)).to_numpy()
    
    complexity = df.iloc[:, 0].values

    return complexity, results
