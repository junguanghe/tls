# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:08:38 2023

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

YMAX = 10
XMAX = 4

loc = './DOSdata/'
loc1 = '../DOSdata/'
# E = [0, .1, .4, 1.2]
# T = [0.1]*4

# E = [0, .4, 1.2]
# T = [0.9]*3

# E = [.4]*5
# T = [.1, .3, .5, .7, .9]

# E = [3, 3, 3, 3, 3]
# T = [.7, .75, .8, .85, .9]

E = [0.4, 1.5]
T = [0.1, 0.1]
Teff = 2

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()

for i in range(len(E)):
    e = E[i]
    t = T[i]
    c = colors[i]
    x = np.loadtxt(loc1 + 'e=' + str(e) + '_t=' + str(t) + '.txt')
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e)+r', $T/T_{c0}=$' + str(t) + r', $T_{eff}/T_{c0}=$' + str(t), 
            color = c, linestyle='--')
    x = np.loadtxt(loc + 'e=' + str(e) + '_t=' + str(t) + '_teff=' + str(Teff) + '.txt')
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e)+r', $T/T_{c0}=$' + str(t) + r', $T_{eff}/T_{c0}=$' + str(Teff), 
            color = c)
    # gap = np.loadtxt('./GapData/e='+str(e)+'.txt')
    # DOfT = interp1d(gap[0], gap[1], fill_value='extrapolate')
    # D = DOfT(t)
    # ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--',
    #           color=plt.gca().lines[-1].get_color())

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c0}$')
plt.tight_layout()
plt.grid(True)
plt.show()