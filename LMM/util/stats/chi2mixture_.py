'''
Adapted code from fastlmm implementation
'''

from __future__ import absolute_import
import scipy as sp
import scipy.stats as st
import scipy.special
import numpy as np
import fastlmm.util.mingrid as mingrid
import pdb
import logging
from six.moves import range
from IPython import embed


class chi2mixture(object):
    '''
    mixture here denotes the weight on the non-zero dof compnent
    '''
    __slots__ = ['scale','dof','mixture','imax','lrt','scalemin','scalemax',
                 'dofmin','dofmax', 'qmax', 'tol', 'isortlrt','qnulllrtsort',
                 'lrtsort','alteqnull','abserr','fitdof']

    def __init__(self, lrt, tol = 0.0, scalemin = 0.1, scalemax = 10.0,
                 dofmin = 0.1, dofmax = 5.0, qmax = None, alteqnull = None, abserr = None, fitdof = None, dof = None):
        '''
        Input:
        lrt             [Ntests] vector of test statistics
        a2 (optional)   [Ntests] vector of model variance parameters
        top (0.0)       tolerance for matching zero variance parameters or lrts
        scalemin (0.1)  minimum value used for fitting the scale parameter
        scalemax (5.0)  maximum value used for fitting scale parameter
        dofmin (0.1)    minimum value used for fitting the dof parameter
        dofmax (5.0)    maximum value used for fitting dof parameter
        qmax (None)      only the top qmax quantile is used for the fit
        '''
        self.lrt = lrt # statistics from the permutations       
        self.alteqnull = alteqnull # index where the lrt = 0
        self.scale = None # scale 
        self.dof = dof # degree of freedom
        self.mixture = None
        self.scalemin = scalemin # minimum value of scale parameter
        self.scalemax = scalemax # maximum value of scale parameter
        self.dofmin = dofmin # minimum value of degree of freedom
        self.dofmax = dofmax # maximum value of degree of freedom
        self.qmax = qmax # how many permutations to use to fit the null distribution
        self.tol = tol # tolerance for the fitting
        self.__fit_mixture()
        self.isortlrt = None
        self.abserr = abserr # absolute error
        self.fitdof = fitdof # fitting the degree of freedom

    def __fit_mixture(self):
        '''
        fit the mixture component
        '''
        if self.tol < 0.0:
            logging.info('tol has to be larger or equal than zero.')
        if self.alteqnull is None:
            self.alteqnull = self.lrt <= 1e-10
            #logging.info("WARNING: alteqnull not provided, so using alteqnull=(lrt==0)")
        if self.mixture is None: # mixture is estimated as the proportion of tests in which the parameter τ = 0
            self.mixture = 1.0 - (sp.array(self.alteqnull).sum()*1.0)/(sp.array(self.alteqnull).shape[0]*1.0)
            # so it's litterally the number of null statistics equal to 0 divided by the total number of 
            # statistics used for the fitting. mixture = 1 - \pi in the paper
        return self.alteqnull, self.mixture
     

    def fit_params_Qreg(self):
        '''
        Fit the scale and dof parameters of the model by minimizing the squared error between
        the model log quantiles and the log P-values obtained on the lrt values.

        Only the top qmax quantile is being used for the fit (self.qmax is used in fit_scale_logP).
        '''
        
        if self.isortlrt is None:
            self.isortlrt = self.lrt.argsort()[::-1] # indexes of the sorted (in descendent order) lrt        
            self.qnulllrtsort = (0.5 + sp.arange(self.mixture*self.isortlrt.shape[0]))/(self.mixture*self.isortlrt.shape[0])   
            # qnulllrtsort contains the expected distribution under the null hypothesis, 
            # which is a uniform distribution on (⁠\pi, 1). Usually the expected distribution
            # of the p-values in gwas is an uniform distribution on (0, 1).
            self.lrtsort = self.lrt[self.isortlrt]      
        

        resmin = [None] # CL says it had to be a list or wouldn't work, even though doesn't make sense
        if self.fitdof: # fit both scale and dof
            def f(x):
                res = self.fit_scale_logP(dof = x)
                if (resmin[0] is None) or (res['mse']<resmin[0]['mse']):
                    resmin[0] = res
                return res['mse']                   
        else: # fit of the scale parameter only
            def f(x): # objective function; case where there's the fit of the scale parameter only           
                scale = x # <- what we want to fit                       
                mse, imax = self.scale_dof_obj(scale, self.dof) # how we calculate the residual and error
                if (resmin[0] is None) or (resmin[0]['mse'] > mse):
                    resmin[0] = { # bookeeping for CL's mingrid.minimize1D
                        'mse':mse,
                        'dof':self.dof,
                        'scale':scale,
                        'imax':imax,
                    }                
                return mse 

        # Actual minimization
        min = mingrid.minimize1D(f = f, nGrid = 10, minval = 0.1, maxval = 10) # why 5? These extremes are for the parameter we are fitting, i.e. the scaling parameter
        self.dof = resmin[0]['dof'] # we keep it fixed to 1
        self.scale = resmin[0]['scale'] # the fitted parameter
        self.imax = resmin[0]['imax'] # the greatest index of the lrt used for the fitting; for us it doesn't 
                                      # make sense, as we use all of them
        # Printing results
        print('fitted dof: ' + str(np.round(self.dof, 2)))
        print('fitted scale: ' + str(np.round(self.scale, 2)))
        return resmin[0]       


    def fit_scale_logP(self, dof = None):        
        '''
        Extracts the top qmax lrt values to do the fit.        
        '''              
      
        if dof is None:
            dof =  self.dof
        resmin = [None]                       
       
        def f(x):            
            scale = x 
            err, imax = self.scale_dof_obj(scale, dof)
            if (resmin[0] is None) or (resmin[0]['mse'] > err):
                resmin[0] = { #bookeeping for CL's mingrid.minimize1D
                    'mse':err,
                    'dof':dof,
                    'scale':scale,
                    'imax':imax,
                }
            return err

        # minimizing the scale parameter given the dof    
        min = mingrid.minimize1D(f = f, nGrid = 10, minval = self.scalemin, maxval = self.scalemax )        
        return resmin[0]

        
    def scale_dof_obj(self, scale, dof):
        '''
        function where it is defined the error function to be minimized,
        that is the difference between:
        - the logarithm (it doesn't matter which base) of the expected p-values, namely 
          the uniform (pi_est, 1), where pi_est is the mixture parameter estimated from 
          the data, as detail at the beginning of this script
        '''            
        base = sp.exp(1) # fitted params are invariant to this logarithm base (i.e. 10 or e)
        nfalse = (len(self.alteqnull) - sp.sum(self.alteqnull)) # number of statistics where the statistics is not 0
                                                                # which is the number of statistics that contribute to
                                                                # the Chi2 with dof = 1; the statistics equal to 0, 
                                                                # instead, contribute to the Chi2 with dof = 0 component
                                                                # of the Chi2 mixture.

        imax = int(sp.ceil(self.qmax*nfalse))  # of only non zero dof component
        # Obtaining the p-values from the statistics under the hypothesized null distribution
        p = st.chi2.sf(self.lrtsort[0:imax]/scale, dof)           
        logp = sp.logn(base, p) # taking the logarithm of the above calculated pvals
        # Calculating the residuals, i.e. calculating the logarithm of the expected distribution of the p-values
        r = sp.logn(base, self.qnulllrtsort[0:imax]) - logp
        # Calculating the error 
        if self.abserr: # absolute error
            err = sp.absolute(r).sum()            
        else: # mean square error
            err = (r*r).mean()     
                     
        return err, imax


    def sf(self, lrt = None, alteqnull = None):
        '''
        compute the survival function of the mixture of scaled chi-square_0 and scaled chi-square_dof
        ---------------------------------------------------------------------------
        Input:
        lrt (optional)     compute survival function for the lrt statistics
                           if None, compute survival function for original self.lrt
        ---------------------------------------------------------------------------
        Output:
        pv                 P-values
        ---------------------------------------------------------------------------
        '''        
        # HERE!!!!!!!!!!!
        lrt =  lrt.astype(float)
        # the Chi2 with dof = 1
        pv = st.chi2.sf(lrt/self.scale, self.dof)*self.mixture 
        # the Chi2 with dof = 0, only for the statistics being 0                       
        pv[sp.array(alteqnull)] = 1.0
        return pv

    