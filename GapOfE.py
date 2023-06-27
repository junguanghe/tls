# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 01:54:19 2023

@author: jungu

this program calculate the gap as a function
of the tunneling spliting E, Delta(E),
where 0 < E < 2 T_c0
"""

from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.optimize import root_scalar
from SelfEnergy import SCE

def DeltaOfE(t, e, x0, x1):
    res = root_scalar(lambda x: SCE(t, e, x), x0 = x0, x1 = x1)
    if not res.converged:
        print('E/Tc0='+str(e)+' not converged')
    return res.root

t = 0.9
E1, E2 = 0, 2
N = 50 # number of points
num_core = cpu_count()

x = np.zeros((2,N))
d0, d1 = 1.4, 1.41 # initial guess
x[0] = np.linspace(E1, E2, N) # x[0][:] is e = E/Tc0
x[1] = Parallel(n_jobs=num_core, verbose=20)(delayed(DeltaOfE)(t, x[0][i], d0, d1)for i in range(N))

loc = './GapData/'
np.savetxt(loc + 't=' + str(t) + '.txt', x)