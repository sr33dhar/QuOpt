#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:14:38 2022

@author: ph30n1x
"""
"""

This code is for Calculating the P = 1 Landscape of MaxCut problems 
that are defined using adjacency matrices C.

Input Needed::
    
    1.) C: The adjacency matrix describing the MaxCut problem instance.
    2.) Instance_list: The list of instances to be studied.
    3.) N: The number of grid points along each gamma or beta axis.
    4.) DList: The list of different bond-dimensions to be studied
    

Output Generated::
    
    1.) Cost_mps: An [N x N] matrix storing the cost values corresponding to each (gamma,beta) for each Dmax studied.
                  This data is saved as a .npy file in the respective instance folders.


"""

import os
import tensornetwork as tn
from Gates import Gates as g
from Expectation import Expectation as exp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tn.set_default_backend("numpy")

#%%


def get_unsymmetric_C(C):
    
    '''
    
    Function that takes an n-qubit Z2 symmetric MaxCut adjacency matrix C as input and returns
    matrices J and h describing a modified n-1 qubit Ising Hamiltonian with broken Z2 symmetry
    by classically fixing the first qubit as '0' and hence first spin == Z_0 as 1.
    
    Input::
        
        C                  : The adjacency matrix describing the q-qubit MaxCut problem instance.
        
    Output::
        
        J                  : The q-1 qubit Interactions matrix.
        h                  : The q-1 qubit field interaction terms.
        
    
    '''
    
    J = 0.5*C[1:,1:]
    h = 0.5*C[0,1:]
    
    return h, J


#%%

def get_Cost_Z2broken(J, h, gamma_beta):
    
    '''
    
    Function that takes in the parameterized state gamma_beta and perturbation constant as
    inputs to return the corresponding cost.
    
    Input::
        
        J                  : The q-1 qubit Interactions matrix.
        h                  : The q-1 qubit field interaction terms.
        gamma_beta         : The parametrized q-1 qubit QAOA state whose cost is to be calculated.
        
    Output::
        
        Cost               : The cost corresponding to the gamma_beta state.
    
    '''

    n = len(h)
    
    Cost = 0
    
    z_i = g.get_Z()
    z_j = g.get_Z()
    
    for i in range(n):
        
        if (h[i] != 0):
            
            g_b = gamma_beta.tensors
            GB = tn.FiniteMPS(g_b, canonicalize = False)
            GB_copy = tn.FiniteMPS(g_b, canonicalize = False)
            
            GB.apply_one_site_gate(z_i,i)
            
            ci = exp.exp_MPS(GB_copy,GB)
            ci = np.real(ci)
            
            Cost = Cost + h[i]*ci
    
    
    for i in range(n-1):
        
        for j in range((n-1),i,-1):
            
            if (J[i,j] != 0):
                
                g_b = gamma_beta.tensors
                GB = tn.FiniteMPS(g_b, canonicalize = False)
                GB_copy = tn.FiniteMPS(g_b, canonicalize = False)
                
                GB.apply_one_site_gate(z_i,i)
            
                GB.apply_one_site_gate(z_j,j)
                
                ci = exp.exp_MPS(GB_copy,GB)
                ci = np.real(ci)
                
                Cost = Cost + J[i,j]*ci        
    
    return Cost


#%%

def get_Cost_mpsIX(C, gamma_beta):
    
    '''
    
    Function that takes in the parameterized state gamma_beta and perturbation constant as
    inputs to return the corresponding cost.
    
    Input::
        
        C                  : The adjacency matrix describing the original q-qubit MaxCut problem instance.
        gamma_beta         : The parametrized q-1 qubit QAOA state whose cost is to be calculated.
        
    Output::
        
        Cost               : The cost corresponding to the gamma_beta state.
    
    '''
    
    n = len(C)
    
    zero = np.array([1.0,0.0])
    
    gamma_beta = tn.FiniteMPS([])
    
    Cost = 0
    
    z_i = g.get_Z()
    z_j = g.get_Z()
       
    
    for i in range(n-1):
        
        for j in range((n-1),i,-1):
            
            if (C[i,j] != 0):
                
                g_b = gamma_beta.tensors
                GB = tn.FiniteMPS(g_b, canonicalize = False)
                GB_copy = tn.FiniteMPS(g_b, canonicalize = False)
                
                GB.apply_one_site_gate(z_i,i)
            
                GB.apply_one_site_gate(z_j,j)
                
                ci = exp.exp_MPS(GB_copy,GB)
                ci = np.real(ci)
                
                Cost = Cost - 0.5*C[i,j]*(1 - ci)        
    
    return Cost
