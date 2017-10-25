# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:21:40 2017

@author: zhaox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import FE_model
import FE_analysis
import uncertainty_analysis


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

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

length = 1000
target = 'E'
# index=list(np.array([3,7,11,15,19])-1)
index = np.arange(21)
#mean = np.ones(len(index)) * 6.3e10
#std = np.ones(len(index)) * 6.3e10 * 0.17
#parm = uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean, std, length)

problem = {
    'num_vars': 21,
    'names': ['x1', 'x2', 'x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
              'x13','x14','x15','x16','x17','x18','x19','x20','x21',],
    'bounds': [[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],
               [-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],
               [-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],
               [-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],[-0.3, 0.3],
               [-0.3, 0.3],]
}

## Generate samples
param_values = saltelli.sample(problem, 100)
parm=(param_values+1)*6.3e10

## Run model (example)
FEM_freq = uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis=analysis1, parm=parm, target='E',index=index)
Y=FEM_freq[:,1]

g = sns.PairGrid(pd.DataFrame(FEM_freq[:,:5]))
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
#Y = FEM_freq[:, 0]
#
## Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
#print(Si['S1'])
