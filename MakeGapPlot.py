#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 01:06:37 2022

@author: jason
"""

import matplotlib.pyplot as plt
import numpy as np

E = [0, 1.2, 2, 3]

loc = './GapData/'

fig, ax = plt.subplots(figsize=(4,3))

for e in E:
    x = np.loadtxt(loc + 'e=' + str(e) + '.txt')
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e))

ax.legend(loc='lower left')
ax.set_ylabel(r'$\Delta/T_{c0}$')
ax.set_xlabel(r'$T/T_{c0}$')
plt.grid(True)
plt.tight_layout()
plt.show()