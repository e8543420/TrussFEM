# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:21:40 2017

@author: zhaox
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

import FE_model
import FE_analysis
import uncertainty_analysis

mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
analysis1.run()

# np.savetxt('test_freq.csv',random_freq,delimiter=',')



# analysis1.plot(mode=11,sf=1.1)

# uncertainty_analysis.uncertainty_analysis.plot_with_ellipse(test_freq[:,[0,2]])

test_freq = np.loadtxt(open('test_freq.csv', 'rb'), delimiter=',', skiprows=0)

length = 100
target = 'E'
# index=list(np.array([3,7,11,15,19])-1)
index = np.arange(21)
mean = np.ones(len(index)) * 6.3e10
std = np.ones(len(index)) * 6.3e10 * 0.17
parm = uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean, std, length)
