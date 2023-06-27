# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:36:30 2022

@author: jungu

This is an old version of the calculation of the finite-Delta self energy
didn't simplify the expression at all, slow as hell
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


# the Fermi-distribution function
def nf(x, t):
    return 1/(np.exp(x/t)+1)

def ne(t, e):
    return nf( e/2 + 1j*np.pi*t/2, t)

def ng(t, e):
    return nf(-e/2 + 1j*np.pi*t/2, t)


# integrand in Sigma_Delta
def funcd(x, D, t, n, e):
    ret = 0
    wn = (2*n+1)*np.pi*t
    sr = np.sqrt(x*x+D*D) # the square-root term in the denominator
    neng = ne(t,e)*ng(t,e) # the term ne*ng
    ret += (neng - ng(t,e) + np.tanh(e/2/t)*nf(-sr,t))/(1j*wn + sr - e)
    ret -= (neng - ng(t,e) + np.tanh(e/2/t)*nf( sr,t))/(1j*wn - sr - e)
    ret += (neng - ne(t,e) - np.tanh(e/2/t)*nf(-sr,t))/(1j*wn + sr + e)
    ret -= (neng - ne(t,e) - np.tanh(e/2/t)*nf( sr,t))/(1j*wn - sr + e)
    ret /= 2*sr # should be a real number
    return np.real(ret)
    

# Sigma_Delta/Delta
def SigmaD(D, t, n, e):
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret -= B1*quad(funcd, -np.inf, np.inf, (D,t,n,e))[0]
    return ret


# integrand in Sigma_omega
def funcw(x, D, t, n, e):
    ret = 0
    wn = (2*n+1)*np.pi*t
    sr = np.sqrt(x*x+D*D) # the square-root term in the denominator
    neng = ne(t,e)*ng(t,e) # the term ne*ng
    ret += (neng - ng(t,e) + np.tanh(e/2/t)*nf(-sr,t))/(1j*wn + sr - e)
    ret += (neng - ng(t,e) + np.tanh(e/2/t)*nf( sr,t))/(1j*wn - sr - e)
    ret += (neng - ne(t,e) - np.tanh(e/2/t)*nf(-sr,t))/(1j*wn + sr + e)
    ret += (neng - ne(t,e) - np.tanh(e/2/t)*nf( sr,t))/(1j*wn - sr + e)
    ret /= 2 # should be an imaginary number
    return np.imag(ret)
    

# i*Sigma_omega / omega_n
def SigmaW(D, t, n, e):
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret -= -B1*quad(funcw, -np.inf, np.inf, (D,t,n,e))[0]/(2*n+1)/np.pi/t
    return ret


# the self-consistent equation
def SCE(t, e):
    ret = 0
    n = 0
    cur = (1 + SigmaD(0, t, n, e))/np.abs(1 + SigmaW(0, t, n, e)) - 1
    cur *= 2/(2*n+1) # even in n, so only sum over n>=0 and multiply by 2
    ret += cur
    while np.abs(cur/ret) > EPSILON:
        n += 1
        cur = (1 + SigmaD(0, t, n, e))/np.abs(1 + SigmaW(0, t, n, e)) - 1
        cur *= 2/(2*n+1) # even in n, so only sum over n>=0 and multiply by 2
        ret += cur
    # print(n)
    return ret - np.log(t)

# solve the self-consistent equation
def SolveSCE(e):
    res = root_scalar(SCE, args=(e,), x0 = x0, x1 = x1)
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

x[1][1:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SolveSCE)(x[0][i])
                                                 for i in range(1,N))

np.savetxt('test_'+'_'.join(str(x) for x in [AKF,NVC,S])+'.txt', x)
