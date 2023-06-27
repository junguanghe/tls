#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:25:20 2022

@author: jason
record the integral error
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from tqdm import tqdm

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

def dos(w, D, t, e):
    SW = SigmaW(D, t, w, e)
    SD = SigmaD(D, t, w, e)
    wtilde = w - SW[0]
    Dtilde = D + SD[0]
    return (np.imag(wtilde/(np.sqrt(Dtilde**2 - wtilde**2))),
            SW[1] + SW[2] + SD[1] + SD[2])

N = 2000 # number of grid points
wrange = 5 # range of omega, i.e. -wrange <= omega <= wrange
e = 1.5 # the energy spliting of the TLS e = E/T_c0
t = 0.1 # the reduced temperature t = T/T_c0
delta = 1e-5 # the small imaginary part in i*omega_n -> omega + i*delta
LIMIT = 5000 # number of subdivisions in scipy quad

gap = np.loadtxt('./GapData/e='+str(e)+'.txt')
DOfT = interp1d(gap[0], gap[1], fill_value='extrapolate')
D = DOfT(t)

x = np.zeros((3,N))
x[0] = np.linspace(-wrange, wrange, N)
for i in tqdm(range(N)):
    x[1][i], x[2][i] = dos(x[0][i]+1j*delta, D,t,e)
    
loc = './DOSdata/'
np.savetxt(loc + 'e=' + str(e) + '_t=' + str(t) + '.txt', x)