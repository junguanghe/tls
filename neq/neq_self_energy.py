# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:20:36 2023

@author: jungu
"""

import numpy as np
from scipy.integrate import quad

EPSILON = 1e-7 # the cutoff for Matsubara sum
AKF = 0.1 # parameter: k_f*a
NVC = 10. # parameter: N(0)*V_0^2*ci/Tc0
S = 0.1 # parameter: the overlap parameter

B0 = 4/15*AKF**4*NVC**2*S**2 # parameter: ci*N(0)*B0/Tc0
B1 = 1/6*AKF**2*NVC**2 # parameter: ci*N(0)*B1/Tc0

Teff = 2

# integrand in Sigma_omega
def funcw(xi, D, t, n, e, pm):
    wn = (2*n+1)*np.pi*t
    PG = np.exp(e/Teff) / (np.exp(e/Teff) + np.exp(-e/Teff))
    PE = 1 - PG
    y = pm*np.sqrt(xi*xi + D*D)
    ret = (1 - (PG - PE)*np.tanh(y/2/t))/(wn*wn + (y-e)**2)
    return ret

# integrand in Sigma_Delta
def funcd(xi, D, t, n, e, pm):
    y = pm*np.sqrt(xi*xi + D*D)
    return funcw(xi, D, t, n, e, pm)*(y-e)/y

# i*Sigma_omega / omega_n
def SigmaW(D, t, n, e):
    PG = np.exp(e/Teff) / (np.exp(e/Teff) + np.exp(-e/Teff))
    PE = 1 - PG
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)*4*PG*PE
    ret += B1*(quad(funcw, -np.inf, 0, args = (D,t,n,e,-1))[0]
               + quad(funcw, 0, np.inf, args = (D,t,n,e,1))[0])
    return ret

# Sigma_Delta/Delta
def SigmaD(D, t, n, e): # should have D>0
    PG = np.exp(e/Teff) / (np.exp(e/Teff) + np.exp(-e/Teff))
    PE = 1 - PG
    ret = B0/np.sqrt((2*n+1)**2*t*t+D*D/np.pi**2)*4*PG*PE
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