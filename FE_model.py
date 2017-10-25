# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:29:12 2017

@author: zhaox
"""

import numpy as np
import copy

class mesh:
    elementNodes=[]
    nodeCoord=[]
    nnode=0
    dof=0
    def __init__(self,length_x=5,length_y=1,elementNodes=None):
        self.def_nodes_retangular(length_x,length_y)
        if elementNodes==None:            
            elementNodes = np.array([[1,2],[1,3],[1,4],[2,4],
                                     [3,4],[3,5],[3,6],[4,6],
                                     [5,6],[5,7],[5,8],[6,8],
                                     [7,8],[7,9],[7,10],[8,10],
                                     [9,10],[9,11],[9,12],[10,12],[11,12]
                                     ]).astype(int)
        self.def_conectivity(elementNodes)      
        self.nnode = self.nodeCoord.shape[0]
        self.dof = self.nodeCoord.shape[1]*self.nnode
    def def_nodes_retangular(self,length_x,length_y):
        el=0
        nodeCoord=np.zeros(((length_x+1)*(length_y+1),2))
        for i in range(0,length_x+1):
            for j in range(0,length_y+1):               
                nodeCoord[el,:]=np.array([i,j])
                el=el+1
        self.nodeCoord = nodeCoord.astype(int)
    def def_conectivity(self,elementNodes):
        elementNodes=np.add(elementNodes,-1) #To deal with index of 0
        self.elementNodes = elementNodes.astype(int)   

class properties:
    E=0
    rho=0
    A=0
    def __init__(self,mesh,E=7e+10,rho=2700,A=0.03):
        self.mesh=mesh
        self.calculate_elem_length()
        self.def_properties(E,rho,A)
    def calculate_elem_length(self):
        elementNodes=self.mesh.elementNodes
        nelm=elementNodes.shape[0]
        nodeCoord=self.mesh.nodeCoord
        L=np.zeros(nelm)
        for e in range(0,nelm):
            ind=elementNodes[e,:]

            xa=nodeCoord[ind[1],0]-nodeCoord[ind[0],0]
            ya=nodeCoord[ind[1],1]-nodeCoord[ind[0],1]
            L[e]=np.sqrt(xa**2+ya**2)
        self.L=L
    def def_properties(self,E,rho,A):
        self.E=E*np.ones(self.L.shape)
        self.rho=rho*np.ones(self.L.shape)
        self.A=A*np.ones(self.L.shape)
    def modify(self,target='E',index=None,data=None):
        """
        #------------Modify properties and reanalysis--------------
        #properties2=properties.modify('E',list(np.array([2,8])-1),np.ones(2)*6.3e10)
        #FE2=FE_model.FE_model(mesh,properties2,BC)
        #analysis2=FE_analysis.modal_analysis(FE2)
        #analysis2.run()
        """
        newself=copy.deepcopy(self)
        if target == 'E' :
            newself.E[index]=data
        elif target == 'rho' :
            newself.rho[index]=data
        elif target == 'A' :
            newself.A[index]=data
        else:
            print ('Warning: Unidentified parameter name! No Change is made!')
        
        return newself
        

class boundary_condition:
    ActiveDof=0
    def __init__(self,mesh,fixedDof=np.array([1,2,3,4])):
        self.fixedDof=fixedDof
        self.deal_ActiveDof(mesh)
    def deal_ActiveDof(self,mesh):
        self.ActiveDof=set(range(0,mesh.dof))-set(self.fixedDof-1)


class FE_model:
    def __init__(self,mesh,properties,boundary_condition):
        self.assemble_matrix(mesh,properties,boundary_condition)
        self.properties=properties
        self.boundary_condition=boundary_condition
        self.mesh=mesh
    def assemble_matrix(self,mesh,properties,boundary_condition):
        nodeCoord=mesh.nodeCoord
        xx=nodeCoord[:,0]
        yy=nodeCoord[:,1]
        nNodes=nodeCoord.shape[0]
        elementNodes=mesh.elementNodes
        nElements=elementNodes.shape[0]
        
        #Element properties
        E=properties.E
        rho=properties.rho
        A=properties.A
        
        ndof=int(mesh.dof/mesh.nnode)        
        neldof=ndof*elementNodes.shape[1]
        GlobalDof=ndof*nNodes
        
        K=np.zeros([GlobalDof,GlobalDof])
        M=np.zeros([GlobalDof,GlobalDof])
        elementDof=np.zeros([nElements,neldof]).astype(int)
        
        #Build system matrices
        for e in range(0,nElements):
            ind=elementNodes[e,:]
            elementDof[e,:]=np.matrix([2*ind[0],2*ind[0]+1,
                                     2*ind[1],2*ind[1]+1])
            xa=xx[ind[1]]-xx[ind[0]]
            ya=yy[ind[1]]-yy[ind[0]]
            Le=np.sqrt(xa**2+ya**2)
            cosa=xa/Le
            sena=ya/Le
            L=np.matrix([[cosa,sena,0,0],[0,0,cosa,sena]])
            
            #mass matrix
            Me=0.5*(rho[e]*A[e]*Le)*np.matrix([[1,0],[0,1]])
            temp_Me=M[elementDof[e,:]][:,elementDof[e,:]]+L.T*Me*L
            for i in range(0,temp_Me.shape[0]):
                for j in range(0,temp_Me.shape[1]):
                    M[elementDof[e,:][i],elementDof[e,:][j]]=temp_Me[i,j]
            #stiffness matrix
            Ke=0.5*(E[e]*A[e]*Le)*np.matrix([[1,-1],[-1,1]])
            temp_Ke=K[elementDof[e,:]][:,elementDof[e,:]]+L.T*Ke*L
            for i in range(0,temp_Ke.shape[0]):
                for j in range(0,temp_Ke.shape[1]):
                    K[elementDof[e,:][i],elementDof[e,:][j]]=temp_Ke[i,j]
        self.M=M
        self.K=K
        self.GlobalDof=GlobalDof
        self.xx=xx
        self.yy=yy
        self.elementDof=elementDof
        self.ActiveDof=boundary_condition.ActiveDof
        self.nodeCoord=mesh.nodeCoord
        self.elementNodes=mesh.elementNodes
        
        