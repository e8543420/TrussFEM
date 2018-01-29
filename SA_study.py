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
random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))

#pd_parm=pd.DataFrame({'Parameter 3':parm_test[:,2],'Parameter 7':parm_test[:,6],'Parameter 15':parm_test[:,14],'Parameter 19':parm_test[:,18]})
#ax1=sns.jointplot('Parameter 3','Parameter 7',data=pd_parm)
#ax2=sns.jointplot('Parameter 15','Parameter 19',data=pd_parm)

#Only list are randomnized
#parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=np.array([1,1,1,1,1])*6.3e10,std=np.ones(5)*6.3e10*0.17,length=100)
#random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array([3,4,5,6,7])-1))

#g = sns.PairGrid(pd.DataFrame(parm_test[:,index]), diag_sharey=False)
#g.map_lower(sns.kdeplot, cmap="Blues_d")
#g.map_upper(plt.scatter)
#g.map_diag(sns.kdeplot, lw=3)

#Initial guess of the parameters

#mean_init_parm=np.ones(21)*7e10
#std_init_parm=np.ones(21)*7e10*0.05
#cov_init_parm=np.diag(std_test_parm**2)
#parm=np.random.multivariate_normal(mean=mean_init_parm,cov=cov_init_parm,size=1000)
#init_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))

np.savetxt('test_freq.csv',random_freq,delimiter=',')

test_freq = np.loadtxt(open('test_freq.csv', 'rb'), delimiter=',', skiprows=0)

#%% Parameter selection

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
if method_flag==1:
    param_values = saltelli.sample(problem, sample_number)
    parm=(param_values+1)*7e10
elif method_flag==2:
    param_values = latin.sample(problem,sample_number)
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
elif method_flag==7:
    param_values = saltelli.sample(problem, sample_number)
    parm=(param_values+1)*7e10    


## Run model (example)
FEM_freq = uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis=analysis1, parm=parm, target='E',index=index)

## Statistical model

##Compare test data and FEM data
#fig, ax = plt.subplots()
#ax.scatter(test_freq[:,3],test_freq[:,5],marker='o',label='Nominal test data')
#ax.scatter(FEM_freq[:1000:10,3],FEM_freq[:1000:10,5],marker='x',label='Initial data')
#ax.legend()


## Single frequency
#Y=FEM_freq[:,0]

## Least square
#mean_test_freq=np.mean(test_freq,axis=0)
#Y=np.zeros(FEM_freq[:,0].shape)
#for i in range(0,20):
#    Y+=((FEM_freq[:,i]-mean_test_freq[i])/mean_test_freq[i])**2
#Y=np.sqrt(Y)

#mean_test_freq=np.mean(test_freq,axis=0)
#Y=np.ones(FEM_freq[:,0].shape)
#for i in range(0,17):
#    rv = multivariate_normal(mean_test_freq[i], np.cov(test_freq[:,i]))
#    y = rv.pdf(FEM_freq[:,i])
#    Y+=y


# PCA projection
#n_component=5
#pca = PCA(n_components=n_component)
#FEM_freq_PCA = pca.fit(FEM_freq).transform(FEM_freq)
#test_freq_PCA = pca.fit(test_freq).transform(test_freq)
##Y = Y[:,0]
#Y=np.linalg.norm(FEM_freq_PCA - test_freq_PCA)


### Propebility
# Multivariate_normal********************

order_invloved=20
test_freq=test_freq[:,:order_invloved]
FEM_freq=FEM_freq[:,:order_invloved]
#init_freq=init_freq[:,:order_invloved]

mean_test=np.mean(test_freq,axis=0)
cov_test=np.cov(test_freq,rowvar=False)
mean_FEM=np.mean(FEM_freq,axis=0)
cov_FEM=np.cov(FEM_freq,rowvar=False)
#mean_init=np.mean(init_freq,axis=0)
#cov_init=np.cov(init_freq,rowvar=False)

test_freq_normalized=np.zeros(test_freq.shape)
FEM_freq_normalized=np.zeros(FEM_freq.shape)
#init_freq_normalized=np.zeros(init_freq.shape)
for i in range(0,order_invloved):
    test_freq_normalized[:,i]=(test_freq[:,i])/mean_test[i]
    FEM_freq_normalized[:,i]=(FEM_freq[:,i])/mean_test[i]
#    init_freq_normalized[:,i]=(init_freq[:,i])/mean_test[i]

mean_test_normalized=np.mean(test_freq_normalized,axis=0)
cov_test_normalized=np.cov(test_freq_normalized,rowvar=False)
mean_FEM_normalized=np.mean(FEM_freq_normalized,axis=0)
cov_FEM_normalized=np.cov(FEM_freq_normalized,rowvar=False)
#mean_init_normalized=np.mean(init_freq_normalized,axis=0)
#cov_init_normalized=np.cov(init_freq_normalized,rowvar=False)

rv = multivariate_normal(mean_FEM_normalized, cov_FEM_normalized)
Y1 = rv.logpdf(FEM_freq_normalized)

rv = multivariate_normal(mean_test_normalized, cov_test_normalized)
Y2 = rv.logpdf(FEM_freq_normalized)

#rv = multivariate_normal(mean_init_normalized, cov_init_normalized)
#Y3 = rv.logpdf(FEM_freq_normalized)

Y=Y2

Y_con=Y1

#list_parm=list()
#for index,y in enumerate(Y):
#    if y > -400:
#        list_parm.append(parm[index,:])
#        Y[index]=1
#    else:
#        Y[index]=0



# KDE 
#mean_test=np.mean(test_freq,axis=0)
#cov_test=np.cov(test_freq,rowvar=False)
#mean_FEM=np.mean(FEM_freq,axis=0)
#cov_FEM=np.cov(FEM_freq,rowvar=False)
#
#test_freq=test_freq[:,:17]
#FEM_freq=FEM_freq[:,:17]
#
#test_freq_normalized=np.zeros(test_freq.shape)
#FEM_freq_normalized=np.zeros(FEM_freq.shape)
#for i in range(0,17):
#    test_freq_normalized[:,i]=(test_freq[:,i]-mean_test[i])/mean_test[i]
#    FEM_freq_normalized[:,i]=(FEM_freq[:,i]-mean_test[i])/mean_test[i]
#
#mean_test_normalized=np.mean(test_freq_normalized,axis=0)
#cov_test_normalized=np.cov(test_freq_normalized,rowvar=False)
#
#kernel = stats.gaussian_kde(test_freq_normalized.T)
#Y=kernel.logpdf(FEM_freq_normalized.T)

## Log function
#mean_test_freq=np.mean(test_freq,axis=0)
#Y=np.zeros(FEM_freq[:,0].shape)
#for i in range(0,20):
#    Y+=np.log(np.abs(((FEM_freq[:,i]-mean_test_freq[i])/mean_test_freq[i])))
#Y=np.abs(Y)




# Draw the scatter of Frequencies
#g = sns.PairGrid(pd.DataFrame(test_freq[:,:]), diag_sharey=False)
#g.map_lower(sns.kdeplot, cmap="Blues_d")
#g.map_upper(plt.scatter)
#g.map_diag(sns.kdeplot, lw=3)
#
## Perform analysis
if method_flag==1:
    Si = sobol.analyze(problem, Y, print_to_console=False)
    figure_keys={'ax1_title':'S1',
                 'ax2_title':'S1_conf',
                 'ax2_lable':'Parameter index',
                 'ax3_title':'ST',
                 'ax4_title':'ST_conf',
                 'ax4_lable':'Parameter index',
                 'ax5_parm':'S2',
                 'ax5_title':'Second order sensitivity',
                 'ax5_lable':'Parameter index',
            }
elif method_flag==2:
    Si = delta.analyze(problem, param_values, Y, num_resamples=10, conf_level=0.95, print_to_console=False)
    figure_keys={'ax1_title':'S1',
                 'ax2_title':'S1_conf',
                 'ax2_lable':'Parameter index',
                 'ax3_title':'delta',
                 'ax4_title':'delta_conf',
                 'ax4_lable':'Parameter index',
            }
    Si_con = delta.analyze(problem, param_values, Y_con, num_resamples=10, conf_level=0.95, print_to_console=False)
    f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
    SS1=(Si_con['S1'][1:])/Si['S1'][1:]
    SS2=(Si_con['delta'][1:])/Si['delta'][1:]
    sns.barplot(np.arange(2,22),np.abs(SS1),ax=ax1)
    sns.barplot(np.arange(2,22),np.abs(SS2),ax=ax2)
    ax1.set_title('SS1')
    ax2.set_title('SDelta')
    ax2.set_xlabel('Sensitivity')        
    
elif method_flag==3:
    Si = dgsm.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False)
    figure_keys={'ax1_title':'dgsm',
                 'ax2_title':'dgsm_conf',
                 'ax2_lable':'Parameter index',
                 'ax3_title':'vi',
                 'ax4_title':'vi_std',
                 'ax4_lable':'Parameter index',
            }
    Si_con = dgsm.analyze(problem, param_values, Y_con, conf_level=0.95, print_to_console=False)
    f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
    SS1=(Si_con['dgsm'][1:])/Si['dgsm'][1:]
    SS2=(Si_con['vi'][1:])/Si['vi'][1:]
    sns.barplot(np.arange(2,22),np.abs(SS1),ax=ax1)
    sns.barplot(np.arange(2,22),np.abs(SS2),ax=ax2)
    ax1.set_title('Sdgsm')
    ax2.set_title('Svi')
    ax2.set_xlabel('Sensitivity')   
    
elif method_flag==4:
    Si = fast.analyze(problem, Y, print_to_console=False)
    figure_keys={'ax1_title':'S1',
                 'ax2_title':'ST',
                 'ax2_lable':'Parameter index',
            }
    Si_con = fast.analyze(problem, Y_con, print_to_console=False)
    f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
    SS1=(np.array(Si_con['S1'][1:]))/np.array(Si['S1'][1:])
    SS2=(np.array(Si_con['ST'][1:]))/np.array(Si['ST'][1:])
    sns.barplot(np.arange(2,22),np.abs(SS1),ax=ax1)
    sns.barplot(np.arange(2,22),np.abs(SS2),ax=ax2)
    ax1.set_title('SS1')
    ax2.set_title('SST')
    ax2.set_xlabel('Sensitivity')   

elif method_flag==5:
    Si = ff_analyze(problem, param_values, Y, second_order=True, print_to_console=False)
    
elif method_flag==6:
    Si = morris.analyze(problem, param_values, Y, conf_level=0.95, 
                    print_to_console=False,
                    num_levels=4, grid_jump=2, num_resamples=100)
    figure_keys={'ax1_title':'mu',
                 'ax2_title':'sigma',
                 'ax2_lable':'Parameter index',
                 'ax3_title':'mu_star',
                 'ax4_title':'mu_star_conf',
                 'ax4_lable':'Parameter index',
            }
elif method_flag==7:
    Si_con = sobol.analyze(problem, Y_con, print_to_console=False)
    Si = sobol.analyze(problem, Y, print_to_console=False)
    figure_keys={'ax1_title':'S1',
                 'ax2_title':'S1_conf',
                 'ax2_lable':'Parameter index',
                 'ax3_title':'ST',
                 'ax4_title':'ST_conf',
                 'ax4_lable':'Parameter index',
                 'ax5_parm':'S2',
                 'ax5_title':'Second order sensitivity',
                 'ax5_lable':'Parameter index',
            }    
    SST=(Si_con['ST'][1:])/Si['ST'][1:]
    SS1=(Si_con['S1'][1:])/Si['S1'][1:]
    
#    f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
#    sns.barplot(np.arange(2,22),np.abs(SST),ax=ax1,color="gray")
#    sns.barplot(np.arange(2,22),np.abs(SS1),ax=ax2,color="gray")
#    ax1.set_title('SST')
#    ax2.set_title('SS1')
#    ax2.set_xlabel('Sensitivity')    
    
    f1,(ax1)=plt.subplots(1,1,sharex=True)
    sns.barplot(np.arange(2,22),np.abs(SST),ax=ax1,color="gray")
    ax1.set_xlabel('Parameter number')    
    ax1.set_ylabel('Composite sensitivity indices') 

# Plot the figure 
f1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
sns.barplot(np.arange(1,22),Si[figure_keys['ax1_title']],ax=ax1)
sns.barplot(np.arange(1,22),Si[figure_keys['ax2_title']],ax=ax2)
ax1.set_title(figure_keys['ax1_title'])
ax2.set_title(figure_keys['ax2_title'])
ax2.set_xlabel(figure_keys['ax2_lable'])

f2,(ax3,ax4)=plt.subplots(2,1,sharex=True)
sns.barplot(np.arange(1,22),Si[figure_keys['ax3_title']],ax=ax3)
sns.barplot(np.arange(1,22),Si[figure_keys['ax4_title']],ax=ax4)
ax3.set_title(figure_keys['ax3_title'])
ax4.set_title(figure_keys['ax4_title'])
ax4.set_xlabel(figure_keys['ax4_lable'])

f3=plt.figure()
ax5=f3.add_axes()
g_S2=sns.heatmap(Si[figure_keys['ax5_parm']],ax=ax5,xticklabels=np.arange(2,22), yticklabels=np.arange(1,21))
g_S2.set_title(figure_keys['ax5_title'])
g_S2.set_xlabel(figure_keys['ax5_lable'])
g_S2.set_ylabel(figure_keys['ax5_lable'])


# Print the first-order sensitivity indices
#print(Si['S1'])
