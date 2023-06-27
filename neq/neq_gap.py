# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:37:51 2023

@author: jungu
"""

from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.optimize import root_scalar
from neq_self_energy import SCE

def DeltaOfT(t, e, x0, x1):
    res = root_scalar(lambda x: SCE(t, e, x), x0 = x0, x1 = x1)
    if not res.converged:
        print('E/Tc0='+str(e)+' not converged')
    return res.root

def TOfDelta(D, e, x0, x1):
    res = root_scalar(lambda x: SCE(x, e, D), x0 = x0, x1 = x1)
    if not res.converged:
        print('E/Tc0='+str(e)+' not converged')
    return res.root

e = 0.4
N = 50 # number of points
num_core = cpu_count()

x = np.zeros((2,N))
d0, d1 = 1.7, 1.71 # initial guess
x[0][:N//2] = np.linspace(0.01, 0.7, N//2) # x[0][:] is t = Tc/Tc0
x[1][:N//2] = Parallel(n_jobs=num_core,
                       verbose=20)(delayed(DeltaOfT)(x[0][i],
                                                     e,
                                                     d0,
                                                     d1)for i in range(N//2))
t0, t1 = 0.9, 0.91 # initial guess
start = x[1][N//2-1]*2 - x[1][N//2-2]
x[1][N//2:] = np.linspace(start, 0.01, N-N//2) # x[1][:] is Delta/Tc0
x[0][N//2:] = Parallel(n_jobs=num_core,
                       verbose=20)(delayed(TOfDelta)(x[1][i],
                                                     e,
                                                     t0,
                                                     t1)for i in range(N//2,N))

                                                     
loc = './GapData/'
Teff = 2 # also set this in neq_self_energy.py
np.savetxt(loc + 'e=' + str(e) + '_Teff=' + str(Teff) + '.txt', x)