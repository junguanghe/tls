# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 02:15:28 2023

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np

loc = './GapData/'

t = 0.7

fig, ax = plt.subplots(figsize=(4,3))

x = np.loadtxt(loc + 't=' + str(t) + '.txt')
ax.plot(x[0], x[1], label=r'$T/T_{c0}=$'+str(t))

ax.legend()
ax.set_ylabel(r'$\Delta/T_{c0}$')
ax.set_xlabel(r'$E/T_{c0}$')
plt.grid(True)
plt.tight_layout()
plt.show()