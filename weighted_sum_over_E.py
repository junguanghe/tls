#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:12:45 2023

@author: jason
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

Tc0 = 9.3
e0 = 23.2/Tc0
J = 2.2/Tc0

Ne = 501
E0,E1 = J+1e-5,10
t = 0.9

wrange = 5
N = 2000

gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
D = DOfE(J)

SWs = np.zeros((Ne, N),dtype=np.complex_)
SDs = np.zeros((Ne, N),dtype=np.complex_)

loc = './DOSdata/SelfEnergyData/'
SWs = np.loadtxt(loc + 'SW' +
           '_Ne=' + str(Ne) +
           '_E0=' + str(round(E0,2)) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D),2)) +
           '_t=' + str(t) + '.txt', dtype=np.complex_)

SDs = np.loadtxt(loc + 'SD' +
           '_Ne=' + str(Ne) +
           '_E0=' + str(round(E0,2)) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D),2)) +
           '_t=' + str(t) + '.txt', dtype=np.complex_)

w = np.linspace(-wrange, wrange, N)
E = np.linspace(E0, E1, Ne)

######################################################## Gaussian distribution

# std = 0.3
# mu = 0.15
# weights = 1/np.sqrt(2*np.pi)/std * np.exp(-((E-mu)/std)**2/2)

# SW = np.average(SWs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)
# SD = np.average(SDs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)

# wtilde = w - SW
# Dtilde = D + SD

# x = np.zeros((2, N))
# x[0] = w
# x[1] = np.imag(wtilde/np.sqrt(Dtilde**2 - wtilde**2))

# loc = './DOSdata/'
# np.savetxt(loc + 'e_averaged' + '_gaussian' +
#            '_std=' + str(std) +
#            '_Ne=' + str(Ne) +
#            '_E0=' + str(E0) +
#            '_E1=' + str(E1) +
#            '_t=' + str(t) + '.txt', x)

######################################################## Gaussian distribution

######################################################## Lorentian distribution

pdf = E/(E**2 + e0**2 - J**2)/np.sqrt(E**2-J**2)
weights = pdf/trapezoid(pdf, E)

SW = np.average(SWs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)
SD = np.average(SDs, axis = 0, weights = weights) # uniform distribution e = U(E0, E1)

wtilde = w - SW
Dtilde = D + SD

x = np.zeros((2, N))
x[0] = w
x[1] = np.imag(wtilde/np.sqrt(Dtilde**2 - wtilde**2))

loc = './DOSdata/'
np.savetxt(loc + 'e_averaged' + '_wipf_model' +
           '_e0=' + str(round(e0,2)) +
           '_J=' + str(round(J,2)) +
           '_Ne=' + str(Ne) +
           '_E0=' + str(E0) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D))) +
           '_t=' + str(t) + '.txt', x)

######################################################## Lorentian distribution