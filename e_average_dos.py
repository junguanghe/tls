#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:13:13 2023

@author: jason
"""

from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
# from tqdm import tqdm

AKF = 0.1 # parameter: k_f*a
NVC = 10. # parameter: N(0)*V_0^2*ci/Tc0
S = 0.1 # parameter: the overlap parameter

B0 = 4/15*AKF**4*NVC**2*S**2 # parameter: ci*N(0)*B0/Tc0
B1 = 1/6*AKF**2*NVC**2 # parameter: ci*N(0)*B1/Tc0

# integrand in Sigma_omega
def funcw(xi, D, t, w, e, pm, ri):
    y = pm*np.sqrt(xi*xi + D*D) # the variable y= +- sqrt(xi**2 + Delta**2)
    ret = (1 - np.tanh(e/2/t)*np.tanh(y/2/t))/(-w*w + (y-e)**2)
    return ret.real if ri else ret.imag

# integrand in Sigma_Delta
def funcd(xi, D, t, w, e, pm, ri):
    y = pm*np.sqrt(xi*xi + D*D)
    return funcw(xi, D, t, w, e, pm, ri)*(y-e)/y

def Dfunc(y,D):
    if np.abs(y) <= D:
        return 0
    return np.abs(y)/np.sqrt(y*y-D*D)

# Sigma_omega
def SigmaW(D, t, w, e):
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    int1 = quad(funcw, -np.inf, 0, args = (D,t,w,e,-1, True), limit=LIMIT)
    int2 = quad(funcw, 0, np.inf, args = (D,t,w,e,1, True), limit=LIMIT)
    ret += B1*( int1[0] + int2[0] )
    ret *= w
    ret += 1j*B1*np.pi/2*(Dfunc( w+e,D)*(1-np.tanh(e/2/t)*np.tanh(( w+e)/2/t))
                         +Dfunc(-w+e,D)*(1-np.tanh(e/2/t)*np.tanh((-w+e)/2/t)))
    return -ret, int1[1], int2[1]

# Sigma_Delta
def SigmaD(D, t, w, e): # should have D>0
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    int1 = quad(funcd, -np.inf, 0, args = (D,t,w,e,-1, True), limit=LIMIT)
    int2 = quad(funcd, 0,  np.inf, args = (D,t,w,e, 1, True), limit=LIMIT)
    ret += B1*( int1[0] + int2[0] )
    ret += 1j*B1*np.pi/2*(
                   Dfunc( w+e,D)*(1-np.tanh(e/2/t)*np.tanh(( w+e)/2/t))/(w+e)
                  +Dfunc(-w+e,D)*(1-np.tanh(e/2/t)*np.tanh((-w+e)/2/t))/(w-e) )
                     # avoid omega = e or -e, may encounter division by zero
    return D*ret, int1[1], int2[1]

def SigmaW_e(e, t, D):
    res = np.zeros(N,dtype=np.complex_)
    err1 = np.zeros(N)
    err2 = np.zeros(N)
    for i,w in enumerate(omega):
        res[i], err1[i], err2[i] = SigmaW(D, t, w+1j*delta, e)
    return res

def SigmaD_e(e, t, D):
    res = np.zeros(N,dtype=np.complex_)
    err1 = np.zeros(N)
    err2 = np.zeros(N)
    for i,w in enumerate(omega):
        res[i], err1[i], err2[i] = SigmaD(D, t, w+1j*delta, e)
    return res

def dos(t, D):
    SWs = np.zeros((Ne, N),dtype=np.complex_)
    SDs = np.zeros((Ne, N),dtype=np.complex_)
    SWs[:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SigmaW_e)(e, t, D)for e in E)
    SDs[:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SigmaD_e)(e, t, D)for e in E)
    
    SW = np.average(SWs, axis = 0) # uniform distribution e = U(E0, E1)
    SD = np.average(SDs, axis = 0) # uniform distribution e = U(E0, E1)
    
    wtilde = omega - SW
    Dtilde = D + SD
    return np.imag(wtilde/np.sqrt(Dtilde**2 - wtilde**2))

Tc0 = 9.3
e0 = 3/Tc0
J = 2.2/Tc0


N = 2000 # number of grid points
wrange = 5 # range of omega, i.e. -wrange <= omega <= wrange
Ne = 501 # number of energy splittings
E0,E1 = J+1e-5,10
t = 0.9 # the reduced temperature t = T/T_c0
delta = 1e-5 # the small imaginary part in i*omega_n -> omega + i*delta
LIMIT = 5000 # number of subdivisions in scipy quad

E = np.linspace(E0, E1, Ne)
omega = np.linspace(-wrange, wrange, N)

num_core = cpu_count()

gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
# D = DOfE((E0+E1)/2)
D = DOfE(J)

x = np.zeros((2, N))
x[0] = omega

SWs = np.zeros((Ne, N),dtype=np.complex_)
SDs = np.zeros((Ne, N),dtype=np.complex_)
SWs[:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SigmaW_e)(e, t, D)for e in E)
SDs[:] = Parallel(n_jobs=num_core, verbose=20)(delayed(SigmaD_e)(e, t, D)for e in E)

loc = './DOSdata/SelfEnergyData/'
np.savetxt(loc + 'SW' +
           '_Ne=' + str(Ne) +
           '_E0=' + str(round(E0,2)) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D),2)) +
           '_t=' + str(t) + '.txt', SWs)

np.savetxt(loc + 'SD' +
           '_Ne=' + str(Ne) +
           '_E0=' + str(round(E0,2)) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D),2)) +
           '_t=' + str(t) + '.txt', SDs)

weights = None
SW = np.average(SWs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)
SD = np.average(SDs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)

wtilde = omega - SW
Dtilde = D + SD

x[1] = np.imag(wtilde/np.sqrt(Dtilde**2 - wtilde**2))

loc = './DOSdata/'
np.savetxt(loc + 'e_averaged' +
           '_Ne=' + str(Ne) +
           '_E0=' + str(E0) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D),2)) +
           '_t=' + str(t) + '.txt', x)