# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:16:04 2022

@author: jungu
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
    sr = pm*np.sqrt(xi*xi + D*D)
    ret = (1 - np.tanh(e/2/t)*np.tanh(sr/2/t))/(-w*w + (sr-e)**2)
    return ret.real if ri else ret.imag

# integrand in Sigma_Delta
def funcd(xi, D, t, w, e, pm, ri):
    sr = pm*np.sqrt(xi*xi + D*D)
    return funcw(xi, D, t, w, e, pm, ri)*(sr-e)/sr

# Sigma_omega
def SigmaW(D, t, w, e):
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    RealPart = B1*(quad(funcw, -np.inf, 0, args = (D,t,w,e,-1, True))[0]
               + quad(funcw, 0, np.inf, args = (D,t,w,e,1, True))[0])
    
    ImagPart = B1*(quad(funcw, -np.inf, 0, args = (D,t,w,e,-1, False))[0]
               + quad(funcw, 0, np.inf, args = (D,t,w,e,1, False))[0])
    ret += RealPart + 1j*ImagPart
    return -w*ret

# Sigma_Delta
def SigmaD(D, t, w, e): # should have D>0
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    RealPart = B1*(quad(funcd, -np.inf, 0, args = (D,t,w,e,-1, True))[0]
               + quad(funcd, 0, np.inf, args = (D,t,w,e,1, True))[0])
    
    ImagPart = B1*(quad(funcd, -np.inf, 0, args = (D,t,w,e,-1, False))[0]
               + quad(funcd, 0, np.inf, args = (D,t,w,e,1, False))[0])
    ret += RealPart + 1j*ImagPart
    return D*ret

def SpectralFunc(xi, w, D, t, e):
    wtilde = w - SigmaW(D, t, w, e)
    Dtilde = D + SigmaD(D, t, w, e)
    GR = -wtilde/(-wtilde**2 + xi*xi + Dtilde**2) # the retarded G
    return -GR.imag/np.pi

def dos(w, D, t, e):
    wtilde = w - SigmaW(D, t, w, e)
    Dtilde = D + SigmaD(D, t, w, e)
    return np.imag(wtilde/(np.sqrt(Dtilde**2 - wtilde**2)))
    # GR = lambda xi: -wtilde/(-wtilde**2 + xi*xi + Dtilde**2) # the retarded G
    # LDOS = lambda xi: -GR(xi).imag/np.pi
    # return quad(LDOS, -np.inf, np.inf)

N = 501 # number of grid points
wrange = 1.5 # range of omega, i.e. -wrange*Delta <= omega <= wrange*Delta
e = 0 # the energy spliting of the TLS e = E/T_c0
t = 0.9 # the reduced temperature t = T/T_c0
delta = 1e-3 # the small imaginary part in i*omega_n -> omega + i*delta

gap = np.loadtxt('./GapData/e='+str(e)+'.txt')
DOfT = interp1d(gap[0], gap[1])
D = DOfT(t)

x = np.zeros((2,N))
x[0] = np.linspace(-wrange*D, wrange*D, N)
for i in tqdm(range(N)):
    x[1][i] = dos(x[0][i]+1j*delta, D, t, e)#[0]
    
loc = './DOSdata/'
np.savetxt(loc + 'e=' + str(e) + '_t=' + str(t) + '.txt', x)