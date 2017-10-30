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
from SALib.analyze import delta
from SALib.analyze import dgsm
from SALib.sample import finite_diff
from SALib.analyze import fast
from SALib.sample import fast_sampler
from SALib.analyze.ff import analyze as ff_analyze
from SALib.sample.ff import sample as ff_sample
from SALib.analyze import morris
from SALib.sample.morris import sample as morris_sample

from sklearn.decomposition import PCA

from scipy.stats import multivariate_normal

# Method used
# 1:Sobol
# 2:delta
# 3:dgsm
# 4:fast
# 5:ff
# 6:morris
method_flag=1
sample_number=1000
upb_search=0.3
lob_search=-0.3


mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
analysis1.run()

# np.savetxt('test_freq.csv',random_freq,delimiter=',')

# analysis1.plot(mode=11,sf=1.1)

# uncertainty_analysis.uncertainty_analysis.plot_with_ellipse(test_freq[:,[0,2]])

parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=np.ones(5)*6.3e10,std=np.ones(5)*6.3e10*0.17,length=100)
random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array([4,8,12,16,20])-1))
np.savetxt('test_freq.csv',random_freq,delimiter=',')

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
    'groups': None,
    'bounds': [[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],
               [lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],
               [lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],
               [lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],[lob_search, upb_search],
               [lob_search, upb_search],]
}

## Generate samples
if method_flag==1 or method_flag==2:
    param_values = saltelli.sample(problem, sample_number)
    parm=(param_values+1)*7e10
elif method_flag==3:
    param_values = finite_diff.sample(problem, sample_number, delta=0.001)
    parm=(param_values+1)*7e10
elif method_flag==4:
    param_values = fast_sampler.sample(problem, sample_number)
    parm=(param_values+1)*7e10
elif method_flag==5:
    param_values = ff_sample(problem)
    parm=(param_values[:,:21]+1)*7e10
elif method_flag==6:
    param_values = morris_sample(problem, N=sample_number, num_levels=4, grid_jump=2, \
                      optimal_trajectories=None)
    parm=(param_values+1)*7e10



## Run model (example)
FEM_freq = uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis=analysis1, parm=parm, target='E',index=index)

## Statistical model

## Single frequency
#Y=FEM_freq[:,0]

## Least square
#mean_test_freq=np.mean(FEM_freq,axis=0)
#Y=np.zeros(FEM_freq[:,0].shape)
#for i in range(0,20):
#    Y+=((FEM_freq[:,i]-mean_test_freq[i])/mean_test_freq[i])**2
#Y=np.sqrt(Y)

## PCA projection
#pca = PCA(n_components=1)
#Y = pca.fit(FEM_freq).transform(FEM_freq)
#Y = Y[:,0]

## Propebility
mean_test=np.mean(test_freq,axis=0)
cov_test=np.cov(test_freq,rowvar=False)
mean_FEM=np.mean(FEM_freq,axis=0)
cov_FEM=np.cov(FEM_freq,rowvar=False)

test_freq_normalized=np.zeros(test_freq.shape)
FEM_freq_normalized=np.zeros(FEM_freq.shape)
for i in range(0,20):
    test_freq_normalized[:,i]=(test_freq[:,i]-mean_test[i])/np.sqrt(cov_test[i,i])
    FEM_freq_normalized[:,i]=(FEM_freq[:,i]-mean_test[i])/np.sqrt(cov_test[i,i])

mean_test_normalized=np.mean(test_freq_normalized,axis=0)
cov_test_normalized=np.cov(test_freq_normalized,rowvar=False)
    
rv = multivariate_normal(mean_test_normalized, cov_test_normalized)
Y = rv.logpdf(FEM_freq_normalized)

## Log function
#mean_test_freq=np.mean(test_freq,axis=0)
#Y=np.zeros(FEM_freq[:,0].shape)
#for i in range(0,20):
#    Y+=np.log(np.abs(((FEM_freq[:,i]-mean_test_freq[i])/mean_test_freq[i])))
#Y=np.abs(Y)





## Draw the scatter of Frequencies
#g = sns.PairGrid(pd.DataFrame(FEM_freq[:,:5]))
#g = g.map_diag(plt.hist)
#g = g.map_offdiag(plt.scatter)
#
## Perform analysis
if method_flag==1:
    Si = sobol.analyze(problem, Y, print_to_console=False)
elif method_flag==2:
    Si = delta.analyze(problem, param_values, Y, num_resamples=10, conf_level=0.95, print_to_console=False)
elif method_flag==3:
    Si = dgsm.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False)
elif method_flag==4:
    Si = fast.analyze(problem, Y, print_to_console=False)
elif method_flag==5:
    Si = ff_analyze(problem, param_values, Y, second_order=True, print_to_console=False)
elif method_flag==6:
    Si = morris.analyze(problem, param_values, Y, conf_level=0.95, 
                    print_to_console=False,
                    num_levels=4, grid_jump=2, num_resamples=100)



# Plot the figure 
f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
sns.barplot(np.arange(1,22),Si['S1'],ax=ax1)
sns.barplot(np.arange(1,22),Si['S1_conf'],ax=ax2)
ax1.set_title('S1')
ax2.set_title('S1_conf')
ax2.set_xlabel('Parameter index')

f2,(ax3,ax4)=plt.subplots(2,1,sharex=True)
sns.barplot(np.arange(1,22),Si['ST'],ax=ax3)
sns.barplot(np.arange(1,22),Si['ST_conf'],ax=ax4)
ax3.set_title('ST')
ax4.set_title('ST_conf')
ax4.set_xlabel('Parameter index')

f3=plt.figure()
ax5=f3.add_axes()
g_S2=sns.heatmap(Si['S2'],ax=ax5,xticklabels=np.arange(1,22), yticklabels=np.arange(1,22))
g_S2.set_title('Second order sensitivity')
g_S2.set_xlabel('Parameter index')
g_S2.set_ylabel('Parameter index')


# Print the first-order sensitivity indices
#print(Si['S1'])
