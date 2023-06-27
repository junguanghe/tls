#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:31:37 2023

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

fig, ax = plt.subplots(figsize=(5,3))

# ##################Lorentzian
# x0 = 1
# gamma = 0.3
# pdf = 1/((E-x0)**2 + gamma**2)
# pdf /= sum(pdf)
# ax.plot(E, pdf, label='Lorentzian({},{})'.format(x0, gamma))
# ################

################## p(E)
Tc0 = 9.3
e0 = 23.2/Tc0
J = 2.2/Tc0

E0,E1 = J+1e-5,3
Ne = 94615
E = np.linspace(E0, E1, Ne)

pdf = 2*e0*E/np.pi/(E**2 + e0**2 - J**2)/np.sqrt(E**2-J**2)
norm = trapezoid(pdf, E)
pdf /= norm
ax.plot(E, pdf, label=r'$\epsilon_0=${},  $J=${})'.format(round(e0,2), round(J,2)))
################

ax.legend()
ax.set_xlabel(r'$E/T_{c0}$')
ax.set_ylabel('normalized pdf')
ax.vlines(x=J, ymin=-0.5, ymax=max(pdf), ls='--',
           color='g')
ax.set_xlim(left=0.22, right=0.3)
ax.set_ylim(0, 50)
plt.tight_layout()
plt.grid(True)