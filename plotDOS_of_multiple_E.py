# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 03:42:35 2023

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

# t = 0.7
# Ne = 501 # number of energy splittings
# E0, E1 = 0, 3 # the energy spliting of the TLS e = E/T_c0

# gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
# DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
# D = DOfE(0.15)

# XMAX = 4
# YMAX = 5

# loc = './DOSdata/'
# x = np.loadtxt(loc + 'e_averaged' +
#            '_Ne=' + str(Ne) +
#            '_E0=' + str(E0) +
#            '_E1=' + str(E1) +
#            '_D=' + str(round(float(D),2)) +
#            '_t=' + str(t) + '.txt')

# fig, ax = plt.subplots(figsize = (5,4))
# ax.plot(x[0], x[1], label=r'$E_j\sim U({},{})$'.format(E0,E1))
# ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--',
#           color=plt.gca().lines[-1].get_color())

# ax.legend()
# ax.legend()
# ax.set_ylim(-0.5, YMAX)
# ax.set_xlim(-XMAX,  XMAX)
# ax.set_ylabel(r'$N/N(0)$')
# ax.set_xlabel(r'$\varepsilon/T_{c0}$')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

################################################################# Gaussian

# t = 0.7
# Ne = 501 # number of energy splittings
# E0, E1 = 0, 3 # the energy spliting of the TLS e = E/T_c0

# gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
# DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
# D = DOfE(0.15)

# XMAX = 4
# YMAX = 5

# std = 0.3
# mu = 0.15

# loc = './DOSdata/'
# x = np.loadtxt(loc + 'e_averaged' + '_gaussian' +
#             '_std=' + str(std) +
#             '_Ne=' + str(Ne) +
#             '_E0=' + str(E0) +
#             '_E1=' + str(E1) +
#             '_t=' + str(t) + '.txt')

# fig, ax = plt.subplots(figsize = (5,4))
# ax.plot(x[0], x[1], label=r'$E_j\sim N({},{}^2)$'.format(mu,std))
# ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--',
#           color=plt.gca().lines[-1].get_color())

# ax.legend()
# ax.legend()
# ax.set_ylim(-0.5, YMAX)
# ax.set_xlim(-XMAX,  XMAX)
# ax.set_ylabel(r'$N/N(0)$')
# ax.set_xlabel(r'$\varepsilon/T_{c0}$')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

##################################################################### Gaussian

#################################################################### Lorentzian

Tc0 = 9.3
e0 = 23.2/Tc0
J = 2.2/Tc0

t = 0.9
Ne = 501 # number of energy splittings
E0,E1 = J+1e-5,10 # the energy spliting of the TLS e = E/T_c0

gap = np.loadtxt('./GapData/t='+str(t)+'.txt')
DOfE = interp1d(gap[0], gap[1], fill_value='extrapolate')
D = DOfE(J)

XMAX = 2
YMAX = 10

loc = './DOSdata/'
x = np.loadtxt(loc + 'e_averaged' + '_wipf_model' +
           '_e0=' + str(round(e0,2)) +
           '_J=' + str(round(J,2)) +
           '_Ne=' + str(Ne) +
           '_E0=' + str(E0) +
           '_E1=' + str(E1) +
           '_D=' + str(round(float(D))) +
           '_t=' + str(t) + '.txt')

fig, ax = plt.subplots(figsize = (5,4))

marker_idx = [i for i in range(len(x[0])) if i % 15 == 0 ]
marker_idx.extend([766, 1233, 860, 1139])
ax.plot(x[0], x[1], label=r'$T/T_{c0}=0.9$', marker='.', markevery=marker_idx)
ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--')
          # color=plt.gca().lines[-1].get_color())

ax.legend()
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c0}$')
plt.tight_layout()
plt.grid(True)

E0,E1 = J+1e-5,3
Ne = 94615
E = np.linspace(E0, E1, Ne)
pdf = E/(E**2 + e0**2 - J**2)/np.sqrt(E**2-J**2)
pdf /= trapezoid(pdf, E)
l,b,w,h = [.25, .5, .45, .35]
sax = fig.add_axes([l,b,w,h])
sax.plot(E, pdf, label=r'$\epsilon_0=${},  $J=${}'.format(round(e0,3), round(J,3)))
sax.set_xlabel(r'$E_j/T_{c0}$')
sax.set_ylabel('normalized pdf')
# sax.vlines(x=J, ymin=-0.5, ymax=max(pdf), ls='--',
#            color='g')
sax.set_xlim(left=0.22, right=0.3)
sax.set_ylim(0, 20)
sax.legend()
plt.tight_layout()
plt.grid(True)

plt.show()

#################################################################### Lorentzian

# x = np.load(loc + 't=' + str(t) + '.npy')
# for y in x[::10]:
#     plt.plot(y[0], y[1])