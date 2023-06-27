# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:15:24 2022

@author: jungu

have to separate the integral interval into four parts in order to use the
weight function in the scipy.quad integral.

can transform back to d xi integral to avoid singularity
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

# the density of state
def DOS(y, D):
    return np.abs(y)/np.sqrt(y*y-D*D)

# the divergent weight subtracted density of state
def wDOS(y, D):
    return np.abs(y)/np.sqrt(np.abs(y) + D)

# integrand in Sigma_omega
def funcw(y, D, t, n, e):
    wn = (2*n+1)*np.pi*t
    ret = DOS(y, D)*(1 - np.tanh(e/2/t)*np.tanh(y/2/t))/(wn*wn + (y-e)**2)
    return ret

# integrand in Sigma_Delta
def funcd(y, D, t, n, e):
    return funcw(y, D, t, n, e)*(y-e)/y

# integrand in Sigma_omega with divergent weight subtracted
def wfuncw(y, D, t, n, e):
    wn = (2*n+1)*np.pi*t
    ret = wDOS(y, D)*(1 - np.tanh(e/2/t)*np.tanh(y/2/t))/(wn*wn + (y-e)**2)
    return ret

# integrand in Sigma_Delta with divergent weight subtracted
def wfuncd(y, D, t, n, e):
    return wfuncw(y, D, t, n, e)*(y-e)/y

# i*Sigma_omega / omega_n
def SigmaW(D, t, n, e):
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret += B1*(quad(funcw, -np.inf, -2*D, args = (D,t,n,e,))[0]
               + quad(funcw, 2*D, np.inf, args = (D,t,n,e,))[0])
    
    ret += B1*(quad(wfuncw, -2*D, -D, args = (D,t,n,e,),
                    weight='alg', wvar=(0, -1/2))[0]
               + quad(wfuncw, D, 2*D, args = (D,t,n,e,),
                      weight='alg', wvar=(-1/2, 0))[0])
    return ret

# Sigma_Delta/Delta
def SigmaD(D, t, n, e): # should have D>0
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret += B1*(quad(funcd, -np.inf, -2*D, args = (D,t,n,e,))[0]
               + quad(funcd, 2*D, np.inf, args = (D,t,n,e,))[0])
    
    ret += B1*(quad(wfuncd, -2*D, -D, args = (D,t,n,e,),
                    weight='alg', wvar=(0, -1/2))[0]
               + quad(wfuncd, D, 2*D, args = (D,t,n,e,),
                      weight='alg', wvar=(-1/2, 0))[0])
    return ret


# the self-consistent equation
def SCE(t, e, D):
    ret = 0
    n = 0
    curSD = SigmaD(D, t, n, e)
    cur = 1 + curSD
    cur /= np.sqrt( (2*n+1)**2*(1+SigmaW(D,t,n,e))**2
                   + D*D/t/t/np.pi**2*(1+curSD)**2 )
    cur -= 1/(2*n+1)
    cur *= 2 # even in n, so only sum over n>=0 and multiply by 2
    ret += cur
    while np.abs(cur/ret) > EPSILON:
        n += 1
        curSD = SigmaD(D, t, n, e)
        cur = 1 + curSD
        cur /= np.sqrt( (2*n+1)**2*(1+SigmaW(D,t,n,e))**2
                       + D*D/t/t/np.pi**2*(1+curSD)**2 )
        cur -= 1/(2*n+1)
        cur *= 2 # even in n, so only sum over n>=0 and multiply by 2
        ret += cur
    # print(n)
    return ret - np.log(t)

# solve the self-consistent equation
def SolveSCE(e, D):
    res = root_scalar(SCE, args=(e, D), x0 = x0, x1 = x1)
    if not res.converged:
        print('E/Tc0='+str(e)+' not converged')
    return res.root

start, end, N = 0, 0.5, 51 # grid points of E/Tc0
x0 = 1.00003 # initial guess
x1 = 1.000037 # initial guess
num_core = cpu_count() # number of cores used

x = np.ndarray((2,N))
x[0] = np.linspace(start, end, N) # x[0][:] is the energy spliting E/Tc0
x[1][0] = 1. # x[1][:] is the critical temperature Tc/Tc0

x[1][1:] = Parallel(n_jobs=num_core,
                    verbose=20)(delayed(SolveSCE)(x[0][i], EPSILON)
                                for i in range(1,N))

np.savetxt('test_'+'_'.join(str(x) for x in [AKF,NVC,S])+'.txt', x)
