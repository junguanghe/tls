# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:42:35 2023

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

E = [0, 1.2, 2]

fig, ax = plt.subplots(figsize=(4,3))

loc1 = '../GapData/'
for e in E:
    x = np.loadtxt(loc1 + 'e=' + str(e) + '.txt')
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e))
    
loc = './GapData/'
for i in range(1, len(E)):
    e = E[i]
    c = colors[i]
    x = np.loadtxt(loc + 'e=' + str(e) + '.txt')
    ax.plot(x[0], x[1], linestyle='--', color=c)
    
ax.legend(loc='lower left')
ax.set_ylabel(r'$\Delta/T_{c0}$')
ax.set_xlabel(r'$T/T_{c0}$')
plt.grid(True)
plt.tight_layout()
plt.show()