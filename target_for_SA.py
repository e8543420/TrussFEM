# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:38:39 2017

@author: zhaox
"""

from __future__ import print_function
from scipy import stats
import numpy as np

import time

import FE_model
import FE_analysis
import uncertainty_analysis

mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
analysis1.run()

parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=np.ones(5)*6.3e10,std=np.ones(5)*6.3e10*0.17,length=100)
random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array([3,7,11,15,19])-1))

time.sleep(1)

parm2=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=np.ones(5)*6.3e10,std=np.ones(5)*6.3e10*0.17,length=100)
random_freq2=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm2,target='E',index=list(np.array([3,7,11,15,19])-1))

ttest=stats.ttest_ind(random_freq,random_freq2)
ks2=stats.ks_2samp(random_freq[:,1],random_freq2[:,1])