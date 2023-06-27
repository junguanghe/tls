# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:39:49 2022

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

YMAX = 10
XMAX = 2

loc = './DOSdata/'
# E = [0, .1, .4, 1.2]
# T = [0.1]*4

# E = [0, .4, 1.2]
# T = [0.9]*3

# E = [.4]*5
# T = [.1, .3, .5, .7, .9]

E = [3, 3, 3, 3, 3]
T = [.7, .75, .8, .85, .9]

fig, ax = plt.subplots()

for e,t in zip(E, T):
    x = np.loadtxt(loc + 'e=' + str(e) + '_t=' + str(t) + '.txt')
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e)+r', $T/T_{c0}=$' + str(t))
            # color = 'C2')
    
    gap = np.loadtxt('./GapData/e='+str(e)+'.txt')
    DOfT = interp1d(gap[0], gap[1], fill_value='extrapolate')
    D = DOfT(t)
    ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--',
              color=plt.gca().lines[-1].get_color())

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(1,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c0}$')
plt.tight_layout()
plt.grid(True)
plt.show()