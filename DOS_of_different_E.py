# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 02:48:44 2023

@author: jungu
"""

from joblib import Parallel, delayed, cpu_count
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

# Sigma_omega
def SigmaW(D, t, w, e):
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    Rint1 = quad(funcw, -np.inf, 0, args = (D,t,w,e,-1,  True), limit=LIMIT)
    Rint2 = quad(funcw, 0,  np.inf, args = (D,t,w,e, 1,  True), limit=LIMIT)
    Iint1 = quad(funcw, -np.inf, 0, args = (D,t,w,e,-1, False), limit=LIMIT)
    Iint2 = quad(funcw, 0,  np.inf, args = (D,t,w,e, 1, False), limit=LIMIT)
    ret += B1*( Rint1[0] + Rint2[0] + 1j*(Iint1[0] + Iint2[0]) )
    return -w*ret, Rint1[1]+Rint2[1], Iint1[1]+Iint2[1]

# Sigma_Delta
def SigmaD(D, t, w, e): # should have D>0
    ret = B0*np.pi/np.sqrt(-w*w+D*D)/np.cosh(e/2/t)**2
    Rint1 = quad(funcd, -np.inf, 0, args = (D,t,w,e,-1,  True), limit=LIMIT)
    Rint2 = quad(funcd, 0,  np.inf, args = (D,t,w,e, 1,  True), limit=LIMIT)
    Iint1 = quad(funcd, -np.inf, 0, args = (D,t,w,e,-1, False), limit=LIMIT)
    Iint2 = quad(funcd, 0,  np.inf, args = (D,t,w,e, 1, False), limit=LIMIT)
    ret += B1*( Rint1[0] + Rint2[0] + 1j*(Iint1[0] + Iint2[0]) )
    return D*ret, Rint1[1]+Rint2[1], Iint1[1]+Iint2[1]

def dos(w, D, t, e):
    SW = SigmaW(D, t, w, e)
    SD = SigmaD(D, t, w, e)
    wtilde = w - SW[0]
    Dtilde = D + SD[0]
    return (np.imag(wtilde/np.sqrt(Dtilde**2 - wtilde**2)),
            SW[1] + SD[1],
            SW[2] + SD[2])

N = 2000 # number of grid points
Ne = 101 # number of energy splitings
E0, E1 = 0, 2 # range of energy splittings
wrange = 5 # range of omega, i.e. -wrange <= omega <= wrange
t = 0.7 # the reduced temperature t = T/T_c0
delta = 1e-3 # the small imaginary part in i*omega_n -> omega + i*delta
LIMIT = 500 # number of subdivisions in scipy quad

E = np.linspace(E0, E1, Ne) # the energy spliting of the TLS e = E/T_c0

num_core = cpu_count()

gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
x = np.zeros((Ne,4,N))

# for j,e in enumerate(E):
#     D = DOfE(e)
#     x[j][0] = np.linspace(-wrange, wrange, N)
#     for i in tqdm(range(N)):
#         x[j][1][i], x[j][2][i], x[j][3][i] = dos(x[j][0][i]+1j*delta, D, t, e)

def wrap(e, t):
    D = DOfE(e)
    ret = np.zeros((4,N))
    ret[0] = np.linspace(-wrange, wrange, N)
    # for i in tqdm(range(N)):
    for i in range(N):
        ret[1][i], ret[2][i], ret[3][i] = dos(ret[0][i]+1j*delta, D, t, e)
    return ret

# for j,e in enumerate(E):
#     x[j] = wrap(e, t)

x[:] = Parallel(n_jobs=num_core, verbose=20)(delayed(wrap)(e, t)for e in E)
        
loc = './DOSdata/'
np.save(loc + 't=' + str(t) + '.npy', x)