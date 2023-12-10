#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Copyright 2023, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on %(date)s

@author: %Minglei Lu
"""

import numpy as np
from numpy.random import randint
from numpy.random import rand
from scipy.interpolate import interp1d
from Model_screen import myModel

def GA_self(Input, Output, v, screen):
    ExpSieves = np.array(Output[['ExpSieves']])
    Expmass = np.array(Output[['Expmass']])  
    
    def objective(x):
        auser_model, fuser_model, p_sieve_model, p_static_cumulative = myModel(Input, Output, v, x[0], x[1], x[2], x[3], x[4], screen=screen)
        f = interp1d(p_sieve_model, p_static_cumulative)
        p_transient_cumulative_point = f(ExpSieves)
        MSE = (np.square(np.subtract(p_transient_cumulative_point,Expmass))).mean()
        return MSE
    
    
    def decode(bounds, n_bits, bitstring):
    	decoded = list()
    	largest = 2**n_bits
    	for i in range(len(bounds)):
    		start, end = i * n_bits, (i * n_bits)+n_bits
    		substring = bitstring[start:end]
    		chars = ''.join([str(s) for s in substring])
    		integer = int(chars, 2)
    		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
    		decoded.append(value)
    	return decoded
    
    # tournament selection
    def selection(pop, scores, k=3):
    	selection_ix = randint(len(pop))
    	for ix in randint(0, len(pop), k-1):
    		if scores[ix] < scores[selection_ix]:
    			selection_ix = ix
    	return pop[selection_ix]
    
    def crossover(p1, p2, r_cross):
    	c1, c2 = p1.copy(), p2.copy()
    	if rand() < r_cross:
    		pt = randint(1, len(p1)-2)
    		c1 = p1[:pt] + p2[pt:]
    		c2 = p2[:pt] + p1[pt:]
    	return [c1, c2]
    
    def mutation(bitstring, r_mut):
    	for i in range(len(bitstring)):
    		if rand() < r_mut:
    			bitstring[i] = 1 - bitstring[i]
    
    def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, stopping_score):
        pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
        best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
        for gen in range(n_iter):
            decoded = [decode(bounds, n_bits, p) for p in pop]
            scores = [objective(d) for d in decoded]
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    if scores[i]>1:
                        print('The {:>1}/{:>1} generation. Mean square error > 100%.'.format(gen+1,n_iter))
                    else:
                        print('The {:>1}/{:>1} generation. Mean square error = {:>5}%.'.format(gen+1,n_iter,np.round(scores[i]*100,decimals=2)))
            if best_eval < stopping_score:
                print("Stopping training. Reached desired score.")
                break
            selected = [selection(pop, scores) for _ in range(n_pop)]
            children = list()
            for i in range(0, n_pop, 2):
                p1, p2 = selected[i], selected[i+1]
                for c in crossover(p1, p2, r_cross):
                    mutation(c, r_mut)
                    children.append(c)
            pop = children
        return [best, best_eval]

    stopping_score = 0.0015
    bounds = [[0, 1], [1e-5, 1], [0, 1], [0, 1], [0, 1]]
    n_iter = 20
    n_bits = 16
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, stopping_score)
    print('Done!')
    decoded = decode(bounds, n_bits, best)
    print('(fmat, xWmmin, gamma, alpha, cdratio) = \n (%s) \n score = %f' % (decoded, score))
    
    # best decoded=[fmat, xWmmin, gamma, alpha, cdratio]
    x = decoded
    return x
