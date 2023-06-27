# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:55:21 2022

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np

N = 10000
size = 20 # font size
a = 2
X = 5
x = np.linspace(-X, X, N)
U = (abs(x*x - a*a) / a / a)**1.5 * 3
A = 2


fig, ax = plt.subplots(figsize = (3.5,3))
ax.plot(x, U, label=r'U(X)', linewidth=4)
# ax.annotate('', xy=(-a,-A/5), xytext=(a,-A/5),
#             arrowprops=dict(arrowstyle='<->'))
# ax.annotate('a', xy=(0,-A*.5), fontsize=size)


# sigma = a/2.5
# x1 = np.linspace(-1.25*X, 1.25*X, N)
# phiL = A/2*np.exp(-(x1+a)**2/sigma**2)
# phiR = A/2*np.exp(-(x1-a)**2/sigma**2)
# ax.plot(x1,phiL, label=r'$\phi_L(X)$')
# ax.plot(x1,phiR, label=r'$\phi_R(X)$')

sigma = a*0.8
x1 = np.linspace(-1.25*X, 1.25*X, N)
phiL = np.exp(-(x1+a)**2/sigma**2)
phiR = np.exp(-(x1-a)**2/sigma**2)
phig = (phiL + phiR)/np.sqrt(2)
phie = (phiL - phiR)/np.sqrt(2) + A
ax.plot(x1,phie, label=r'$\phi_e(X)$')
ax.plot(x1,phig, label=r'$\phi_g(X)$')
ax.annotate('', xy=(0,A/5), xytext=(0,A),
            arrowprops=dict(arrowstyle='<->'))
ax.annotate(r'$J$', xy=(0.1,A*.5), fontsize=size)

ax.annotate('', xy=(-a,A/5), xytext=(a,A/5),
            arrowprops=dict(arrowstyle='<->'))
ax.annotate(r'$a$', xy=(0,0), fontsize=size)

ax.set_xlabel('X')
ax.grid(True)
ax.legend()
ax.set_ylim([0,5])
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.set_tight_layout(True)