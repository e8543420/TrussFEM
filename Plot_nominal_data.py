# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:15:22 2018

@author: zhaox
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:21:40 2017

@author: zhaox
"""
#%% Initialization

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
from SALib.sample import latin

from sklearn.decomposition import PCA

from scipy.stats import multivariate_normal
from scipy import stats

# Method used
# 1:Sobol
# 2:delta
# 3:dgsm
# 4:fast
# 5:ff
# 6:morris
method_flag=7
sample_number=100
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

#%% Sampling for test data
#Whole list are randomnized
index=list(np.array([3,7,11,15,19])-1)
mean_test_parm=np.ones(21)*7e10
mean_test_parm[index]=np.ones(5)*6.3e10

std_test_parm=np.ones(21)*7e10*0.05
std_test_parm[index]=np.ones(5)*7e10*0.17
cov_test_parm=np.diag(std_test_parm**2)
cov_test_parm[2,6]=(7e10*0.17)**2
cov_test_parm[6,2]=(7e10*0.17)**2
cov_test_parm[14,18]=-(7e10*0.13)**2
cov_test_parm[18,14]=-(7e10*0.13)**2


parm=np.random.multivariate_normal(mean=mean_test_parm,cov=cov_test_parm,size=100)
#parm=stats.multivariate_normal(mean_test_parm,np.diag(std_test_parm**2)).rvs(size=100)
#parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=mean_test_parm,std=std_test_parm,length=100)
parm_test=parm
cov_parm_test=np.cov(parm_test,rowvar=False)
test_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))


index=list(np.array([3,7,11,15,19])-1)
mean_test_parm=np.ones(21)*7e10

std_test_parm=np.ones(21)*7e10*0.05
cov_test_parm=np.diag(std_test_parm**2)


parm=np.random.multivariate_normal(mean=mean_test_parm,cov=cov_test_parm,size=100)
#parm=stats.multivariate_normal(mean_test_parm,np.diag(std_test_parm**2)).rvs(size=100)
#parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=mean_test_parm,std=std_test_parm,length=100)
parm_FEM=parm
cov_parm_test=np.cov(parm_test,rowvar=False)
FEM_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))


#Compare test data and initial nominal

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(parm_FEM[:,2],parm_FEM[:,4],marker='x',label='Nominal data')
ax1.scatter(parm_test[:,2],parm_test[:,4],marker='o',label='Test data')
ax1.set_xlabel('Parameter 3')
ax1.set_ylabel('Parameter 5')
#ax1.axis('equal')
ax1.legend()


ax2.scatter(FEM_freq[:,0],FEM_freq[:,1],marker='x',label='Nominal data')
ax2.scatter(test_freq[:,0],test_freq[:,1],marker='o',label='Test data')
ax2.set_xlabel('1st modal frequency(Hz)')
ax2.set_ylabel('2nd modal frequency(Hz)')
#ax2.axis('equal')
ax2.legend()



