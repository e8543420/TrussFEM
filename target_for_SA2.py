# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:40:55 2017

@author: zhaox
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = norm.stats(moments='mvsk')

x = np.linspace(norm.ppf(0.001),norm.ppf(0.999), 100)
 

rv = norm()
f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
ax1.plot(x, rv.logpdf(x), 'k-', lw=2, label='frozen logpdf')
ax1.set_title('Log-PDF of multivariate normal distribution')

ax2.plot(x, rv.pdf(x), 'k-', lw=2)
ax2.set_title('Normal PDF of multivariate normal distribution')
