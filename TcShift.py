#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:31:40 2022

@author: jason
"""

from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

EPSILON = 1e-7 # the cutoff for Matsubara sum
AKF = 0.1 # parameter: k_f*a
NVC = 10. # parameter: N(0)*V_0^2*ci/Tc0
S = 0.1 # parameter: the overlap parameter

B0 = 4/15*AKF**4*NVC**2*S**2 # parameter: ci*N(0)*B0/Tc0
B1 = 1/6*AKF**2*NVC**2 # parameter: ci*N(0)*B1/Tc0

# integrand in Sigma_Delta
# will encounter dividing by 0 at x=-e, but never happens in scipy quad()
def funcd(x, t, n, e): 
    return x/(x+e)*np.tanh((x+e)/2/t)/(((2*n+1)*np.pi*t)**2+x*x)

# Sigma_Delta/Delta
def SigmaD(t, n, e): # only takes in n>=0
    ret = B0/(2*n+1)/t/np.cosh(e/2/t)**2\
        +B1*(2*n+1)*t/(e*e/np.pi**2+(2*n+1)**2*t*t)
    ret -= B1*np.tanh(e/2/t)*quad(funcd, -np.inf, np.inf, (t,n,e))[0]
    return ret

# integrand in Sigma_omega
def funcw(x, t, n, e):
    return np.tanh((x+e)/2/t)/(((2*n+1)*np.pi*t)**2+x*x)

# i*Sigma_omega / omega_n
def SigmaW(t, n, e): # only takes in n>=0
    ret = B0/(2*n+1)/t/np.cosh(e/2/t)**2 + B1/(2*n+1)/t
    ret -= B1*np.tanh(e/2/t)*quad(funcw, -np.inf, np.inf, (t,n,e))[0]
    return ret

# the self-consistent equation
def SCE(t, e):
    ret = 0
    n = 0
    cur = (1 + SigmaD(t, n, e))/np.abs(1 + SigmaW(t, n, e)) - 1
    cur *= 2/(2*n+1) # even in n, so only sum over n>=0 and multiply by 2
    ret += cur
    while np.abs(cur/ret) > EPSILON:
        n += 1
        cur = (1 + SigmaD(t, n, e))/np.abs(1 + SigmaW(t, n, e)) - 1
        cur *= 2/(2*n+1) # even in n, so only sum over n>=0 and multiply by 2
        ret += cur
    # print(n)
    return ret - np.log(t)

# solve the self-consistent equation
def SolveSCE(e):
    res = root_scalar(SCE, args=(e,), x0 = x0, x1 = x1)
    if not res.converged:
        print('E/Tc0='+str(e)+' not converged')
    # print(e, res.root)
    return res.root

start, end, N = 0, 0.5, 51 # grid points of E/Tc0
x0 = 1.00003 # initial guess
x1 = 1.000037 # initial guess
num_core = cpu_count() # number of cores used

x = np.ndarray((2,N))
x[0] = np.linspace(start, end, N) # x[0][:] is the energy spliting E/Tc0
x[1][0] = 1. # x[1][:] is the critical temperature Tc/Tc0

# x[1][1:] = np.array(list(map(SolveSCE, x[0][1:])))
x[1][1:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SolveSCE)(x[0][i])
                                                 for i in range(1,N))
# for i in range(1,N):
#     x[1][i] = SolveSCE(x[0][i])
#     # x0, x1 = x1, x[1][i] # 13% faster

loc = './TcData/'
np.savetxt(loc + '_'.join(str(x) for x in [AKF,NVC,S])+'.txt', x)


# e = 0.05 # parameter: E/Tc0
# res = root_scalar(SCE, args=(e), x0 = 1.0001, x1=1.0002)
# print(res)