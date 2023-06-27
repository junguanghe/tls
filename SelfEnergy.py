# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:46:20 2022

@author: jungu

change to d xi integration to avoid singularities of the density of state
at +Delta and -Delta
"""

# from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.integrate import quad
# from scipy.optimize import root_scalar

EPSILON = 1e-7 # the cutoff for Matsubara sum
AKF = 0.1 # parameter: k_f*a
NVC = 10. # parameter: N(0)*V_0^2*ci/Tc0
S = 0.1 # parameter: the overlap parameter

B0 = 4/15*AKF**4*NVC**2*S**2 # parameter: ci*N(0)*B0/Tc0
B1 = 1/6*AKF**2*NVC**2 # parameter: ci*N(0)*B1/Tc0

# integrand in Sigma_omega
def funcw(xi, D, t, n, e, pm):
    wn = (2*n+1)*np.pi*t
    y = pm*np.sqrt(xi*xi + D*D)
    ret = (1 - np.tanh(e/2/t)*np.tanh(y/2/t))/(wn*wn + (y-e)**2)
    return ret

# integrand in Sigma_Delta
def funcd(xi, D, t, n, e, pm):
    y = pm*np.sqrt(xi*xi + D*D)
    return funcw(xi, D, t, n, e, pm)*(y-e)/y

# i*Sigma_omega / omega_n
def SigmaW(D, t, n, e):
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret += B1*(quad(funcw, -np.inf, 0, args = (D,t,n,e,-1))[0]
               + quad(funcw, 0, np.inf, args = (D,t,n,e,1))[0])
    return ret

# Sigma_Delta/Delta
def SigmaD(D, t, n, e): # should have D>0
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)/np.cosh(e/2/t)**2
    ret += B1*(quad(funcd, -np.inf, 0, args = (D,t,n,e,-1))[0]
               + quad(funcd, 0, np.inf, args = (D,t,n,e,1))[0])
    return ret


# the self-consistent equation
# Delta can't be 0 or too small
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

'''
This part below is for checking the above code with TcShift.py
'''
# # solve the self-consistent equation: t(Delta, E), to compare with TcShift
# def SolveSCE(e, D):
#     res = root_scalar(SCE, args=(e, D), x0 = x0, x1 = x1)
#     if not res.converged:
#         print('E/Tc0='+str(e)+' not converged')
#     return res.root

# start, end, N = 0, 0.5, 51 # grid points of E/Tc0
# x0 = 1.00003 # initial guess
# x1 = 1.000037 # initial guess
# num_core = cpu_count() # number of cores used

# x = np.ndarray((2,N))
# x[0] = np.linspace(start, end, N) # x[0][:] is the energy spliting E/Tc0
# x[1][0] = 1. # x[1][:] is the critical temperature Tc/Tc0

# x[1][1:] = Parallel(n_jobs=num_core,
#                     verbose=20)(delayed(SolveSCE)(x[0][i], 0.1)
#                                 for i in range(1,N))
# loc = './TcData/'
# np.savetxt(loc + 'D=0.1_'+'_'.join(str(x) for x in [AKF,NVC,S])+'.txt', x)
