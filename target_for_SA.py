# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:38:39 2017

@author: zhaox
"""
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

test_freq = np.loadtxt(open('test_freq.csv', 'rb'), delimiter=',', skiprows=0)


mean_test=np.mean(test_freq,axis=0)
cov_test=np.cov(test_freq,rowvar=False)
test_freq_normalized=np.zeros(test_freq.shape)
for i in range(0,20):
    test_freq_normalized[:,i]=(test_freq[:,i]-mean_test[i])/cov_test[i,i]

mean_test_normalized=np.mean(test_freq_normalized,axis=0)
cov_test_normalized=np.cov(test_freq_normalized,rowvar=False)

#x, y = np.mgrid[0.7*mean_test[0]:1.3*mean_test[0]:10j, 0.7*mean_test[1]:1.3*mean_test[1]:10j]
#pos = np.dstack((x, y))
#x=test_freq[:,0]
#y=test_freq[:,1]
#pos=test_freq

prop = multivariate_normal.pdf(test_freq_normalized,mean_test_normalized, cov_test_normalized)
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.contourf(x, y, rv.pdf(pos))
#prop=rv.pdf(test_freq)

#g=sns.PairGrid(pd.DataFrame(test_freq))
#g.map_lower(sns.kdeplot, cmap="Blues_d")
#g.map_upper(plt.scatter)
#g.map_diag(sns.kdeplot, lw=3)