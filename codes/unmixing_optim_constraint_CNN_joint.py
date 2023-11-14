
import time
import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy
import os
import matplotlib.cm as cm
PATH_DATA = '../data/' # CHANGE THAT
PATH = '../codes/'
sys.path.insert(1,PATH)

import sys
sys.path.insert(1,PATH_DATA)

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
vcol = ['mediumseagreen','crimson','steelblue','darkmagenta','burlywood','khaki','lightblue','darkseagreen','deepskyblue','forestgreen','gold','indianred','midnightblue','olive','orangered','orchid','red','steelblue']

import IAE_CNN_TORCH_Oct2023 as cnn 
import torch

from general import divergence, NNPU, NMF_fixed_a



def BCD(y,X0,a0,list_model,X=None,estimed_aMVP=1,UpdateA=False,
         step_size_BSP=1e-3,tol=1e-4,niter_max_BSP=1000,niter_max_out=100,tol_BSP=1e-8,norm='1',optim=0):
    """
    Estimate X and a based on BCD using IAE 
    Parameters
    ----------
    y: mesured spectrum
    X0 : initial spectral signature
    X : input spectral signature, for NMSE computation purposes
    a0 : initial mixing weight
    list_model: list of pre-trained IAE models
    estimed_aMVP : estimation of MVP, 1 : yes, 0: no
    step_size_BSP: learning rate in BSP
    tol: tolerance in the outer loop
    niter_max_BSP : maximum number of iterations for inner loop when estimating X
    niter_max_out : maximum number of external iterations
    tol_in: tolerance in the inner loop
    norm: norm to normalize data
    optim: solver used in BSP
    UpdateA: also adjust the mixing weight a when estimate X 
    """
    ####  initial outer loop
    err=1
    ite=0
    itp=0
    ak=a0.copy()
    Xk=X0.copy()
    ## loss
    loss=[divergence(X0,y,a0)] 
    ## NMSE
    if X is not None:
        NMSE_list=[-10*np.log10((np.sum((X0[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0)))] 
    else:
        NMSE_list = None
    ## initial value of lambda
    Lambda_list = [0 for r in range(len(a0)-1)]
    #initial value of a
    Amp_list=[0 for  r in range(len(a0)-1)]
    OldErrIn = 1e32
    OldLoss = 1e32
    I = np.linspace(0,len(a0)-1,len(a0)).astype('int32')

    #### while loop
    while (ite<niter_max_out) & (err>tol) :

        ### Estimate the mixing weight
        
        ak_p = NNPU(y,Xk,ak,estimed_aMVP)

        ### initial inner loop

        Xp = Xk.copy()

        ### use joint IAE model
        if len(list_model)<(len(a0)-1):  ## only one IAE model -> joint 
            
            d=np.shape(list_model[0].anchorpoints)[1]
            Bg = Xp[:,0].dot(ak_p[0])# Bkg* its counting
            if ite<1:
                # iteration 0: no lambda value, estimate X by NMF then use Fast Interpolation to give a initial value of lambda
                tmp=NMF_fixed_a(y,X0,ak_p)
                Lambda0=list_model[0].fast_interpolation((tmp)[np.newaxis,:,1:])["Lambda"][0][0].detach().numpy()
                ## Use barycentric_span_projection to estimate lambda (or X)
                rec = barycentric_span_projection(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[0], Lambda0=Lambda0,Amp0=None, a0=ak_p[1:], UpdateA=UpdateA ,niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm='1')
                # save the estimated lambda in the lambda list and use it as the initial value for the next iteration
                Lambda_list[0] = rec['Lambda']
                Amp_list[0]=rec['Amplitude']
            else:
                # iteration > 0
                rec = barycentric_span_projection(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[0], Lambda0=Lambda_list[0],Amp0=Amp_list[0], a0=ak_p[1:], UpdateA=UpdateA ,niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm='1')
                Lambda_list[0] = rec['Lambda']
                Amp_list[0]=rec['Amplitude']
                #print( rec['Lambda'])
            summ = rec["XRec"].squeeze()
            # update X
            Xp[:d,1:]=summ
            Xp[:,1:]/=np.sum(Xp[:,1:],axis=0)# normalize
         
        ## use individual IAE model    
        else:    
            
            for i in range(np.shape(Xk)[1]-1):
                if (ak_p[i+1] > 1e-10) or (ite ==0):
                    J = I[I != (i+1)]# all radionculides except i
                    Bg = Xp[:,J].dot(ak_p[J]) # spectral signature * a of all radionuclides except i
                    d=np.shape(list_model[i].anchorpoints)[1]# dimension of spectral signature i
                    if ite<1:
                        rec = barycentric_span_projection(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[i], Lambda0=None,Amp0=None, a0=ak_p[i+1]*np.ones((1,)), UpdateA=UpdateA ,niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm='1')
                        Lambda_list[i] = rec['Lambda']
                        Amp_list[i]=rec['Amplitude']
                    else:

                        rec = barycentric_span_projection(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[i], Lambda0=Lambda_list[i],Amp0=None, a0=ak_p[i+1]*np.ones((1,)), UpdateA=UpdateA ,niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm='1')
                            
                        Lambda_list[i] = rec['Lambda']
                        Amp_list[i]=rec['Amplitude']
                        #print( rec['Lambda'])
                    
                    summ = rec["XRec"].squeeze()
                    Xp[:d,i+1]=summ#/np.sum(summ)# normalize X
                    Xp[:,i+1]/=np.sum(Xp[:,i+1])
                    
        # stopping criterion            
        errA= np.mean(np.linalg.norm(ak_p-ak)/np.linalg.norm(ak+1e-10))
        errX= np.mean(np.linalg.norm(Xp-Xk)/np.linalg.norm(Xk+1e-10))
        err=np.maximum(errA,errX)
        if X is not None:# calculate NMSE of estimated X
            NMSE_list.append(-10*np.log10((np.sum((Xp[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0))))

        # Update variables
        Xk=Xp.copy()
        ak=ak_p.copy()

        cost=divergence(Xk,y,ak)
        loss+=[cost]

        ite+=1
        itp+=1
        
        OldErrIn = err
        OldLoss = cost


        cmd = 'loss: '+str(cost)+' / ErrX: '+str(errX)+' / ErrA: '+str(errA)+ ' /: step: '+str(step_size_BSP)
        print('iteration outer: ',ite,' / ',cmd)
            
    if len(list_model)<(len(a0)-1):  ### joint IAE model
        return Xk,ak,np.array(loss),np.array(NMSE_list),np.array(Lambda_list[0]).squeeze()
    else:
        return Xk,ak,np.array(loss),np.array(NMSE_list),np.array(Lambda_list).squeeze()

###############################################################################################################################################
# Parameter fitting
###############################################################################################################################################





def _get_barycenter(Lambda,amplitude=None,model=None,fname=None):

    """
    Reconstruct a barycenter from Lambda
    Parameters
    ----------
    Lambda: lambda used to reconstruct the barycenter
    amplitude: amplitude of X, if: None -> the vector 1
    model: IAE model
    fname: name of IAe model if model is not provided
    
    """

    if model is None:
        model = load_model(fname)
    model.nneg_output=True

    PhiE,_ = model.encode(model.anchorpoints)
    
    
    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(Lambda.astype("float32")), PhiE[model.NLayers-r-1]))
    
    if  amplitude is None:
        tmp=model.decode(B).detach().numpy()
        return tmp
    else:
        tmp=torch.einsum('ijk,i -> ijk',model.decode(B),torch.as_tensor(amplitude.astype("float32"))).detach().numpy()
        return tmp



def barycentric_span_projection(y, Bg = None, model=None, Lambda0=None, Amp0=None,a0=None, tole=1e-4,UpdateA=False ,
                                niter=100, optim=None, norm='1',step_size=1e-3):
    """
    Estimate X (or lambda) using SLSQP 
    Parameters
    ----------
    y: mesured spectrum
    Bg : terms fixed
    model: pre-trained IAE model
    Lambda0: initial value of lambda
    Amp0 : initial amplitude if updateA is True
    a0 : mixing weight (fixed)
    UpdateA: also adjust the mixing weight a when estimate X 
    tol: tolerance in the outer loop
    niter : maximum number of iterations
    tole: tolerance
    norm: norm to normalize data
    optim: solver used in BSP
    step_size: Step size used for numerical approximation of the Jacobian

    """
        
    from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)
    r=0
    PhiE,_ = model.encode(model.anchorpoints) # encode anchor points
    # shape and init values
    d = model.anchorpoints.shape[0]
    b,tx,_ = y.shape
    ty=len(a0)
    loss_val = []
    a = a0
    # init lambda
    if Lambda0 is None:
        if len(a0)==1: # the counting of X has dimension one -> individual IAE model
            #initial lambda is calculated by fast_interpolation
            Lambda = model.fast_interpolation(y-Bg,Amplitude=a0[:,np.newaxis])["Lambda"][0].detach().numpy()
            Lambda=Lambda.squeeze()
        else:# joint IAE model, set lambda = lambda of first anchor point
            Lambda=np.zeros(d)
            Lambda[0]=1
    else:
        Lambda = Lambda0
    # init amplitude 
    if Amp0 is None:
        Amp= np.ones(ty)
    else:
        Amp=Amp0
    # init value of parameters to be estimated: amplitude (if updateA is True) and lambda    
    x0=np.append(Amp,Lambda)    
    # define simplex constraint    
    def simplex_constraint(param):
        Lamb=param[ty:]
        return np.sum(Lamb)-1
    # define the limits values of lambda
    bnds=(model.bounds)# limit values of lambda obtained in training
    for i in range(ty):
        bnds=((0,None),)+bnds
    # define constraints for SLSQP   
    constraints=[{'type': 'eq','fun':simplex_constraint}]
    # barycenter function: provide X with given lambda
    def Func(P):
        P=torch.tensor(P.astype('float32'))
        P = P.reshape(b,-1)
        B = []
        Lambda=P[:,ty:]
  
        for r in range(model.NLayers):
            B.append(torch.einsum('ik,kjl->ijl',Lambda , PhiE[model.NLayers-r-1]))# calculate barycenter
        
        # decode barycenter 
        XRec=model.decode(B)
        #XRec=XRec*(XRec>0)
        XRec=torch.einsum('ijk,ik-> ijk',XRec,1/torch.sum(XRec,axis=(1)))
        if UpdateA: # update X by X* estimated amplitude 
            return torch.einsum('ijk,ik -> ijk',XRec,P[:,0:ty]).detach().numpy()
        else:
            return XRec.detach().numpy()

    # cost function
    def get_cost(param,arg):
        
        y,Bg,a0=arg# recover the parameters in arguments
        XRec = Func(param) # recover X using param
        
        XRec=np.einsum('ijk,ilk->ijl', XRec, a0[np.newaxis, np.newaxis, :])# X*a
        Tot = XRec+Bg
        Tot = Tot*(Tot > 0)+1e-6
        #Tot = Tot*(Tot > 0)+1e-10
        return np.sum(Tot-y*np.log(Tot))# negative log likelihood
        
    sol = minimize(get_cost,x0=x0,args=[y,Bg,a0],constraints=constraints,  
                       bounds=bnds,method='SLSQP',tol=tole,options={'maxiter':niter,'eps':step_size})
 
    param =sol.x# solution obtained by solver
    Amp=param[:ty]
    Lambda=param[ty:]
    
    if len(Lambda.shape)==1:
            Lambda=Lambda[np.newaxis,:]
    Params={}
    Params["Lambda"] = Lambda
    Params['XRec']=_get_barycenter(Lambda,model=model)
    Params['Amplitude']=Amp
    return Params






























# def barycentric_span_projection2(x, Bg = None, model=None, Lambda0=None, a0=None, tole=1e-4,UpdateA=False ,
#                                 niter=100, optim=None, norm='1',constr=True,step_size=1e-3):
#     from scipy.optimize import minimize

#     if model is None:
#         model = load_model(fname)
#     r=0
#     PhiE,_ = model.encode(model.anchorpoints)
#     d = model.anchorpoints.shape[0]
#     b,tx,ty = x.shape

#     # Initialize Lambda0

#     loss_val = []

#     if a0 is None:
#         _,a = _normalize(x,norm=model.normalisation)
#     else:
#         a = a0

#     if Lambda0 is None:
#         if len(a0)>1:
#             Lambda=[1,0,0]
#         else:
#             Lambda = model.fast_interpolation(x-Bg,Amplitude=a0)["Lambda"][0].detach().numpy()
#     else:
#         Lambda = Lambda0
   
#     def Func(P):
#         #print(P)
#         P = P.reshape(b,-1)
#         #print(P.shape)
#         B = []
#         #Lambda=P[:,1::]
        
#         for r in range(model.NLayers):
#             B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(P[:,:-1].astype("float32"))   , PhiE[model.NLayers-r-1]))
#             #B.append(torch.einsum('ik,kjl->ijl',Lambda , PhiE[model.NLayers-r-1]))
#         return model.decode(B).detach().numpy()
#         #return np.einsum('ijk,i -> ijk',model.decode(B).detach().numpy(),P[:,0])   
    
#     def simplex_constraint(param):
#         Lamb=param[:]

#         return np.sum(Lamb)-1
    
#     bnds=(model.bounds)
#     for i in range(ty):
#         bnds=bnds+((0,None),)
    
#     constraints=[{'type': 'eq','fun':simplex_constraint}]
#     #bnds = ((0, None), (0, None),(None,None))
    
#     def get_cost(param,arg):
#         #Lamb, Amplitude= param
#         Lamb=param[:-1]
#         Amp=param[-1:]
#         X,Bg,Amp0=arg
#         if len(Lamb.shape)==1:
#             Lamb=Lamb[np.newaxis,:]
        
        
#         XRec = Func(param)
#         XRec=XRec*(XRec>0)
        
#         if UpdateA:
#             #if len(
            
#             XRec=np.einsum('ijk,ilk->ijl', XRec, Amp0[np.newaxis, np.newaxis, :])
#             #XRec = Amp[:, np.newaxis, np.newaxis] * XRec
#             XRec = Amp * XRec
#         else:
#             XRec = Amp0[:, np.newaxis, np.newaxis] * XRec

#         ### NOT QUITE CLEAN

#         Tot = XRec+Bg
#         Tot = Tot*(Tot > 0)+1e-6
       
#         #Tot = Tot*(Tot > 0)+1e-10
#         return np.sum(Tot-X*np.log(Tot)-X+X*np.log(X))

#     if optim==1:
# #         sol = minimize(get_cost,x0=np.append(Lambda,1),args=[X,Bg,Amplitude],constraints=constraints,  
# #                        bounds=bnds,method='trust-constr',tol=1e-8,options={'maxiter':100})
#         sol = minimize(get_cost,x0=np.append(Lambda,1),args=[x,Bg,a], method='BFGS',tol=tole, options={'maxiter':niter},jac='3-point')
#     else:   
        
#         sol = minimize(get_cost,x0=np.append(Lambda,1),args=[x,Bg,a],constraints=constraints,  
#                        bounds=bnds,method='SLSQP',tol=tole,options={'maxiter':niter,'eps':step_size})
 
#     param =sol.x
#     print(param)
#     print(sol)
#     Lambda=param[:-1]
#     a=param[-1:]*a
#     if len(Lambda.shape)==1:
#             Lambda=Lambda[np.newaxis,:]
#     Params={}
#     Params["Lambda"] = Lambda
#     #Params['XRec'] = get_barycenter(model,Params["Lambda"], Amplitude)
#     Params['XRec']=_get_barycenter(Lambda,model=model)
#     #Params['XRec']=_get_barycenter(Lambda,amplitude=a,model=model)

#     Params['Amplitude']=a
#     #print(sol)
#     return Params

    
    
    

