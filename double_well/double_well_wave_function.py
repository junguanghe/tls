# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:56:02 2023

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import eigh_tridiagonal

U = 5
N = 500
Y = 5
y = np.linspace(-Y, Y, 2*N + 1)
h = y[1] - y[0]
d = U*(y*y - 1)**2 + 2/h/h
e = -np.ones(2*N)/h/h

w, v = eigh_tridiagonal(d, e)
v /= np.sqrt(h)

A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
print(np.allclose(A @ v - v @ np.diag(w), np.zeros((2*N + 1, 2*N + 1))))

print(np.allclose(np.sum(v**2*h, axis=0), np.ones(2*N + 1)))

fig, ax = plt.subplots(5,1,figsize=(4,12))
ax[0].plot(y, U*(y*y-1)**2)
ax[0].set_ylim([0,3*U])
ax[1].plot(y, v[:,0], label='ground state')
ax[1].plot(y, v[:,1], label='first excited state')
V0 = v[:,0]**2*U*(y*y - 1)**2
V1 = v[:,1]**2*U*(y*y - 1)**2
K0 = w[0]*v[:,0]**2 - V0
K1 = w[1]*v[:,1]**2 - V1
EV0 = trapezoid(V0, y)
EV1 = trapezoid(V1, y)
EK0 = trapezoid(K0, y)
EK1 = trapezoid(K1, y)
ax[2].plot(y, V0, label=r'$\langle U(x)\rangle_0=$'+'{:.2f}'.format(EV0))
ax[2].plot(y, V1, label=r'$\langle U(x)\rangle_1=$'+'{:.2f}'.format(EV1))
ax[3].plot(y, K0, label=r'$\langle -\frac{\hbar^2\partial_x^2}{2m}\rangle_0=$'+'{:.2f}'.format(EK0))
ax[3].plot(y, K1, label=r'$\langle -\frac{\hbar^2\partial_x^2}{2m}\rangle_1=$'+'{:.2f}'.format(EK1))
ax[4].plot(y, V0 + K0, label=r'$E_0=$'+'{:.2f}'.format(w[0]))
ax[4].plot(y, V1 + K1, label=r'$E_1=$'+'{:.2f}'.format(w[1]))
# ax[2,0].plot(y, (2*v[:,0]-np.append(v[1:,0],[0])-np.append([0],v[:-1,0]))*v[:,0]/h/h)
# ax[2,1].plot(y, (2*v[:,1]-np.append(v[1:,1],[0])-np.append([0],v[:-1,1]))*v[:,1]/h/h)
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)
ax[4].grid(True)
ax[0].set_title(r'$U(x)/(\frac{\hbar^2}{2ma^2})$')
ax[1].set_title(r'$\psi(x)$')
ax[2].set_title(r'$\psi^*(x)U(x)\psi(x)/(\frac{\hbar^2}{2ma^2})$')
ax[3].set_title(r'$\psi^*(x)(-\frac{\hbar^2\partial_x^2}{2m})\psi(x)/(\frac{\hbar^2}{2ma^2})$')
ax[4].set_title(r'$\psi^*(x)H(x)\psi(x)/(\frac{\hbar^2}{2ma^2})$')
ax[4].set_xlabel('x/a')
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
fig.set_constrained_layout(True)