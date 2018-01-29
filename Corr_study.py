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


from scipy.stats import multivariate_normal
from scipy import stats




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
#cov_test_parm[2,6]=(7e10*0.17)**2
#cov_test_parm[6,2]=(7e10*0.17)**2
#cov_test_parm[14,18]=-(7e10*0.13)**2
#cov_test_parm[18,14]=-(7e10*0.13)**2


parm=np.random.multivariate_normal(mean=mean_test_parm,cov=cov_test_parm,size=1000)
#parm=stats.multivariate_normal(mean_test_parm,np.diag(std_test_parm**2)).rvs(size=100)
#parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=mean_test_parm,std=std_test_parm,length=100)
parm_test=parm
cov_parm_test=np.cov(parm_test,rowvar=False)
random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))

#pd_parm=pd.DataFrame({'Parameter 3':parm_test[:,2],'Parameter 7':parm_test[:,6],'Parameter 15':parm_test[:,14],'Parameter 19':parm_test[:,18]})
#ax1=sns.jointplot('Parameter 3','Parameter 7',data=pd_parm)
#ax2=sns.jointplot('Parameter 15','Parameter 19',data=pd_parm)

np.savetxt('test_freq.csv',random_freq,delimiter=',')

test_freq = np.loadtxt(open('test_freq.csv', 'rb'), delimiter=',', skiprows=0)

#%%  Draw correlation diagram
index=np.arange(0,21)

pd_data = pd.DataFrame(np.hstack((parm[:,index],test_freq)))
col_name = [('p'+str(x+1)) for x in index]+[('f'+str(x)) for x in range(1,21)]
pd_data.columns=col_name
corr = pd_data.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 14))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,annot=True)

cov=pd_data.cov().iloc[21:,:21]
sns.heatmap(cov)



