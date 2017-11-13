# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:53:28 2017

@author: zhaox
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2,f_oneway

class uncertainty_analysis:
    def __init__(self,analysis):
        pass
    def random_series_generator(length=100,mu=1,sigma=0.1,seed=None):
        random.seed(seed)
        x=np.empty(length)
        for i in range(0,length):
            x[i]=random.gauss(mu,sigma) 
        return x
        #plt.hist(x)
    
    def random_two_generator(mean=[0,0],cov=[[1, 0], [1, 1]],size=5000):
        x, y = np.random.multivariate_normal(mean, cov, 5000).T
        return x,y
    def plot_point_cov(points, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma ellipse based on the mean and covariance of a point
        "cloud" (points, an Nx2 array).
        Parameters
        ----------
            points : An Nx2 array of the data points.
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
        Returns
        -------
            A matplotlib ellipse artist
        Example
        -------
        #points = np.random.multivariate_normal(
        #        mean=(1,1), cov=[[0.1, 0], [.1, .1]], size=1000
        #        )
        #
        #x, y = points.T
        #plt.scatter(x, y, color='red')
        #
        #
        #uncertainty_analysis.plot_point_cov(points, nstd=2, alpha=0.5,fill=False,color='blue')
        #
        ##uncertainty_analysis.plot_chi_cov_ellipse(cov,pos,nsig=2,fill=False, color='green')
        #plt.show()
        """
        pos = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        #return uncertainty_analysis.plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
        return uncertainty_analysis.plot_chi_cov_ellipse(cov,pos,nsig=nstd,ax=ax,**kwargs)
       
    def plot_chi_cov_ellipse(cov,pos, q=None, nsig=None, ax=None, **kwargs):
        """
        Parameters
        ----------
        cov : (2, 2) array
            Covariance matrix.
        q : float, optional
            Confidence level, should be in (0, 1)
        nsig : int, optional
            Confidence level in unit of standard deviations. 
            E.g. 1 stands for 68.3% and 2 stands for 95.4%.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
        Returns
        -------
            A matplotlib ellipse artist
        """
        if ax is None:
            ax = plt.gca()
            
        if q is not None:
            q = np.asarray(q)
        elif nsig is not None:
            q = 2 * norm.cdf(nsig) - 1
        else:
            raise ValueError('One of `q` and `nsig` should be specified.')
        r2 = chi2.ppf(q, 2)
    
        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        theta = np.degrees(np.arctan2(*vec[::-1, 0]))
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
        ax.add_artist(ellip)
        return ellip   
    
    def random_parm_generator(mean,std,length):
        parm=np.zeros((length,len(mean)))
        for i in range(0,len(mean)):
            parm[:,i]=uncertainty_analysis.random_series_generator(length=length,mu=mean[i],sigma=std[i],seed=(i+1)*datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        return parm
    
    def random_freq_run(analysis,parm,target,index):
        if analysis.freq is None:
            analysis.run()       
        num_random=parm.shape[0]
        random_freq=np.zeros((num_random,len(analysis.freq)))
        for i in range(0,num_random):
            freq,modn=analysis.reanalysis(target,index,parm[i])
            random_freq[i]=freq.T
        return random_freq
    
    def plot_with_ellipse(random_freq,nstd=2,alpha=0.5):              
        """
        #-----------Generate and plot random frequencies----------------
        #parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=np.ones(5)*6.3e10,std=np.ones(5)*6.3e10*0.17,length=1000)
        #random_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array([3,7,11,5,19])-1))
        #
        #fig,axes =plt.subplots(nrows=1,ncols=2,figsize=(12,5))
        #
        #points=random_freq[:,[0,1]]
        #x, y = points.T
        #uncertainty_analysis.uncertainty_analysis.plot_point_cov(points, nstd=2, alpha=0.5,fill=False,color='blue',ax=axes[0])
        #axes[0].scatter(x, y, color='red')
        #axes[0].set_xlabel('Freq1(Hz)')
        #axes[0].set_ylabel('Freq2(Hz)')
        #
        #points=random_freq[:,[0,2]]
        #x, y = points.T
        #uncertainty_analysis.uncertainty_analysis.plot_point_cov(points, nstd=2, alpha=0.5,fill=False,color='blue',ax=axes[1])
        #axes[1].scatter(x, y, color='red')
        #axes[1].set_xlabel('Freq1(Hz)')
        #axes[1].set_ylabel('Freq3(Hz)')
        #plt.show()
         """
        axes =plt.axes()
        points=random_freq
        x, y = points.T
        uncertainty_analysis.plot_point_cov(points, nstd,axes, alpha=alpha,fill=False,color='blue')
        axes.scatter(x, y, color='red')
        axes.set_xlabel('Frequency(Hz)')
        axes.set_ylabel('Frequency(Hz)')
    
    def six_level_F_test(analysis,target_parm,list_parm_level=1+np.array([-.1,-.05,-.01,.01,.05,.1]),mean=np.ones(1)*7e10,std=np.ones(1)*7e10*0.27,length=100):
        """
        f_all=np.zeros([20,21])
        p_all=np.zeros([20,21])
        for i in range(0,21):
            target_parm=i+1
            f,p=uncertainty_analysis.uncertainty_analysis.six_level_F_test(analysis1,target_parm,list_parm_level=1+np.array([-.1,-.05,-.01,.01,.05,.1]),
                                 mean=np.ones(1)*7e10,std=np.ones(1)*7e10*0.27,length=100)
            f_all[:,i]=f
            p_all[:,i]=p
        plt.matshow(p_all)
        plt.ylabel('Modal order')
        plt.xlabel('Parameter number')
        plt.title('F-test result(p-value)')
        """
        
        parm=uncertainty_analysis.random_parm_generator(mean,std,length)
        random_freq=uncertainty_analysis.random_freq_run(analysis,parm,target='E',index=list(np.array([target_parm])-1))
        sample_freq=np.zeros([len(list_parm_level),random_freq.shape[0],random_freq.shape[1]])
        for i,parm_level in enumerate(list_parm_level) :
            parm2=uncertainty_analysis.random_parm_generator(mean*parm_level,std,length)
            sample_freq[i]=uncertainty_analysis.random_freq_run(analysis,parm2,target='E',index=list(np.array([target_parm])-1))        
        #one way F-test
        #In situation of multi parameters, try http://blog.csdn.net/hjh00/article/details/48783631
        f,p=f_oneway(random_freq,sample_freq[0],sample_freq[1],sample_freq[2],sample_freq[3],sample_freq[4],sample_freq[5])
        return f,p
    def global_sensitivity_LS(analysis,test_freq,parm,sensi_parm_index,index,target='E'):  
        '''
        test_freq=np.loadtxt(open('test_freq.csv','rb'),delimiter=',',skiprows=0)
        length=100
        target='E'
        #index=list(np.array([3,7,11,15,19])-1)
        index=np.arange(21)
        mean=np.ones(len(index))*6.3e10
        std=np.ones(len(index))*6.3e10*0.17
        parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean,std,length)
        
        ss=np.zeros((21))
        for i in range(21):
            sensi_parm_index=i
            ss[i]=uncertainty_analysis.uncertainty_analysis.global_sensitivity_LS(analysis1,test_freq,parm,sensi_parm_index,index,target)
        
        '''
        length_parm=len(parm)
        mean=np.mean(parm,axis=0)
        parm=np.concatenate((parm,mean.reshape(1,len(mean))),axis=0)
        
        f0=0
        D=0
        D_i=0
        for i in range(length_parm):
            parm_i=parm[i]
            fx_i,modn=analysis.reanalysis(target,index,parm_i)
            parm_i=parm[i+1]
            parm_i[sensi_parm_index]=parm[i][sensi_parm_index]
            fx__i,modn=analysis.reanalysis(target,index,parm_i)
            f0=f0+np.sqrt(np.sum(np.square(fx_i-np.mean(test_freq,axis=0))))
            D=D+np.sum(np.square(fx_i-np.mean(test_freq,axis=0)))
            D_i=D_i+np.sqrt(np.sum(np.square(fx_i-np.mean(test_freq,axis=0))))*np.sqrt(np.sum(np.square(fx__i-np.mean(test_freq,axis=0))))
        f0=f0/length_parm
        D=D/length_parm-np.square(f0)
        D_i=D_i/length_parm-np.square(f0)
        SS=1-(D_i/D)
        return SS
    def global_sensitivity_seprate(analysis,test_freq,parm,sensi_parm_index,index,target='E'):        
        '''
        Calculate the sensitivity of different frequencies seprately
        
        test_freq=np.loadtxt(open('test_freq.csv','rb'),delimiter=',',skiprows=0)
        length=100
        target='E'
        #index=list(np.array([3,7,11,15,19])-1)
        index=np.arange(21)
        mean=np.ones(len(index))*6.3e10
        std=np.ones(len(index))*6.3e10*0.17
        parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean,std,length)
        
        ss=np.zeros((21,test_freq.shape[1]))
        for i in range(21):
            sensi_parm_index=i
            ss[i]=uncertainty_analysis.uncertainty_analysis.global_sensitivity_seprate(analysis1,test_freq,parm,sensi_parm_index,index,target)
        
        '''
        length_parm=len(parm)
        mean=np.mean(parm,axis=0)
        parm=np.concatenate((parm,mean.reshape(1,len(mean))),axis=0)
        
        f0=np.zeros(test_freq.shape[1])
        D=np.zeros(test_freq.shape[1])
        D_i=np.zeros(test_freq.shape[1])
        for i in range(length_parm):
            parm_i=parm[i]
            fx_i,modn=analysis.reanalysis(target,index,parm_i)
            parm_i=parm[i+1]
            parm_i[sensi_parm_index]=parm[i][sensi_parm_index]
            fx__i,modn=analysis.reanalysis(target,index,parm_i)
            f0=f0+(fx_i-np.mean(test_freq,axis=0))
            D=D+np.square(fx_i-np.mean(test_freq,axis=0))
            D_i=D_i+(fx_i-np.mean(test_freq,axis=0))*(fx__i-np.mean(test_freq,axis=0))
        f0=f0/length_parm
        D=D/length_parm-np.square(f0)
        D_i=D_i/length_parm-np.square(f0)
        SS=1-(D_i/D)
        return SS





