# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:51:56 2022

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np

AKF = 0.1
NVC = 10.
S = 0.1

loc = './TcData/'
x = np.loadtxt(loc + '_'.join(str(x) for x in [AKF,NVC,S])+'.txt')

fig, ax = plt.subplots()
ax.plot(x[0], x[1], label=r'$c_iN(0)V_0^2/T_{c0}=$'+str(NVC))
ax.legend()
ax.set_ylabel(r'$T_c/T_{c0}$')
ax.set_xlabel(r'$E/T_{c0}$')
plt.tight_layout()
plt.show()