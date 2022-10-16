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


##############################
## Parameter Initialization ##
##############################

q = 12 # number of Qubits

l1 = int(np.floor((q)/2))   # Paramteres required for the SWAP network
l2 = int(np.floor((q-1)/2)) # Paramteres required for the SWAP network

N = 10 # Number of sampling points in Gamma and Beta
pi = np.pi

Gamma = [0.0 + x*pi/(N-1) for x in range(N)] # List of 100 Gamma angles between [0 , pi]
Beta = [0.0 + x*0.50*pi/(N-1) for x in range(N)]  # List of 100 Beta angles between [0 , pi/2]

G,B = np.meshgrid(Beta,Gamma) # Creating a mesh for the 3D plots

# DList = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2] # List of Bond-dimensions to be simulated
DList = [64] # List of Bond-dimensions to be simulated

folder_location = os.path.dirname(os.path.realpath(__file__)) # Location of the MPS_QAOA Folder
local_location = '/QAOA/MaxCut/Erdos/MxC_Q'+str(q)+'/Q' #Location of files within the MPS_QAOA Folder
location = folder_location + local_location

tag = 'R' # R for Random

D_highest = int(2**np.floor(q/2)) # Highest possible bond-dimension for a q-qubit MPS

Normalize = False # Normalization constraint for the MPS-QAOA states. IMPORTANT: Keep False

Pert_const = 0.0 # A perturbation constant added to break the Z2 symmetry.

Instance_list = [0] # The indices of the instances one wishes to run the calculations for. 
                    # Eg: If q = 12, enter [0,42] if the instances of interest are 12R0 and 12R42






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
            
            Cost = Cost - h[i]*ci
    
    
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
                
                Cost = Cost - J[i,j]*ci        
    
    return Cost


#%%

def get_Cost_Z2symmetric(C, gamma_beta):
    
    '''
    
    Function that takes in the parameterized state gamma_beta and perturbation constant as
    inputs to return the corresponding cost.
    
    Input::
        
        C                  : The adjacency matrix describing the original q-qubit MaxCut problem instance.
        gamma_beta         : The parametrized q-1 qubit QAOA state whose cost is to be calculated.
        
    Output::
        
        Cost               : The cost corresponding to the original MaxCut problem with |0> tensor |gamma,beta>.
    
    '''
    
    n = len(C)
    
    Zero = np.array([[[1],
                      [0]]],
                    dtype = np.complex128)
    
    GB_list = gamma_beta.tensors.copy() # Copying as otherwise gamma_beta state is altered by insert.
    gamma_beta = tn.FiniteMPS(GB_list.insert(0,Zero), canonicalize=False)
    
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



#%%


def QAOA_gamma_block(gamma_beta, gamma, J, h, Dmax):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer.
    
    Input::
        
        gamma_beta  : An existing QAOA state upon which one applies the Gamma layer
        gamma       : The free parameter that needs to be optimized
        J           : The Interaction Energies matrix describing the EC3 problem instance.
        h           : The self-energy matrix describing the EC3 problem instance.
        Dmax        : The maximum bond-dimension limit imposed on the MPSs
        
    Output::
        
        gamma_beta  : The QAOA state with an additional Cost layer of the QAOA added to it
    
    '''
    
    n = len(h)
    
    gamma_beta = tn.FiniteMPS(gamma_beta.tensors, canonicalize=False)
    gamma_beta.canonicalize(normalize=Normalize)
    
    Q_ord = np.linspace(start = 0, stop = (n-1), num = n, dtype = int)
    
    SWAP = g.get_SWAP()
    
    for i in range(n):
    
        Rz = g.get_Rz(2*gamma*h[i])
        gamma_beta.apply_one_site_gate(Rz, i)
    
    
    for i in range(n):
        
        if (i < (n-1)):
            
            for k in range(n-1):
                
                if (Q_ord[k] < Q_ord[k+1]):
                    
                    Jij = g.get_Jij(gamma, J[Q_ord[k]][Q_ord[k+1]])
                    
                    gamma_beta.position(site=k, normalize=Normalize)
                    gamma_beta.apply_two_site_gate(Jij, site1 = k, site2 = (k+1),
                                               max_singular_values=Dmax, center_position=k)
                    
        
        if (i%2 == 0):
            
            for s in range(l1):
                
                Q_ord[2*s],Q_ord[2*s+1] = Q_ord[2*s+1],Q_ord[2*s]
                
                gamma_beta.position(site=(2*s), normalize=Normalize)
                gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s), site2 = (2*s+1),
                                               max_singular_values=Dmax, center_position=(2*s))
                
            
        else:
            
            for s in range(l2):
                
                Q_ord[2*s+1],Q_ord[2*s+2] = Q_ord[2*s+2],Q_ord[2*s+1]
                
                gamma_beta.position(site=(2*s+1), normalize=Normalize)
                gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s+1), site2 = (2*s+2),
                                               max_singular_values=Dmax, center_position=(2*s+1))
        
    gamma_beta = gamma_beta.tensors[::-1]
    gamma_beta = [gamma_beta[x].transpose([2,1,0]) for x in range(n)]
    gamma_beta = tn.FiniteMPS(gamma_beta, canonicalize = False)
    
    return gamma_beta

#%%


def QAOA_beta_block(gamma_beta, beta):
    
    '''
    
    Function that takes in the parameter beta and applies
    a single layer of the mixing block within a single QAOA layer.
    
    Input::
        
        gamma_beta: An existing QAOA state upon which one applies the Gamma layer
        beta: The free parameter that needs to be optimized
        
    Output::
        
        gamma_beta: The QAOA state with an additional mixing layer of the QAOA added to it
    
    '''
    
    n = len(gamma_beta.tensors)
    
    gamma_beta = tn.FiniteMPS(gamma_beta.tensors, canonicalize=False)
    
    Rx = g.get_Rx(2*beta)
        
    for i in range(n):
        
        gamma_beta.apply_one_site_gate(Rx, i)
    
    return gamma_beta


#%%
