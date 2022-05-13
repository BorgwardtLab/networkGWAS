'''
Adapted code from fastlmm implementation
'''

from __future__ import absolute_import
import scipy as sp
import scipy.stats as st
import scipy.special
import numpy as np
import pdb
import logging
from six.moves import range
from IPython import embed
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

class chi2mixture_zero(object):
    '''
    mixture here denotes the weight on the non-zero dof compnent
    '''
    __slots__ = ['scale','dof','mixture','imax','lrt','scalemin','scalemax',
                 'dofmin','dofmax', 'qmax', 'tol', 'isortlrt','qnulllrtsort',
                 'lrtsort','alteqnull','abserr','fitdof']

    def __init__(self, lrt, tol = 0.0, qmax = None, alteqnull = None, abserr = None):
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
        self.mixture = None
        self.qmax = qmax # how many permutations to use to fit the null distribution
        self.tol = tol # tolerance for the fitting
        self.isortlrt = None
        self.abserr = abserr # absolute error


    def fit_mixture(self):
        '''
        this function is for calculating the fitting for different values of the threshold 
        defined for the amplitude of the Chi2 dof = 0 component.
        '''
        power = np.arange(-15, -4)
        threshold_vect = []
        for p in power:
            threshold_vect.append( 10**float(p))
        
        mse_vec, res_vec = [], {}
        
        for i, thr in enumerate(threshold_vect):
            self.alteqnull = self.lrt <= thr
            # mixture is estimated as the proportion of tests in which the parameter τ = 0
            self.mixture = 1.0 - (sp.array(self.alteqnull).sum()*1.0)/(sp.array(self.alteqnull).shape[0]*1.0)
            # so it's litterally the number of null statistics equal to 0 divided by the total number of 
            # statistics used for the fitting. mixture = 1 - \pi in the paper
            res, mse = self.fit_params() # fitting!
            mse_vec.append(mse)
            res_vec[i] = [self.mixture, res]
        
        
        idx_min = np.argmin(mse)
        return res_vec[idx_min], mse_vec[idx_min], threshold_vect[idx_min]
     

    def fit_params(self):
        '''
        Fit the scale and dof parameters of the model by minimizing the squared error between
        the model log quantiles and the log P-values obtained on the lrt values.

        Only the top qmax quantile is being used for the fit (self.qmax is used in fit_scale_logP).
        '''
        
        if self.isortlrt is None:
            self.isortlrt = self.lrt.argsort()[::-1] # indexes of the sorted (in descendent order) lrt        
            self.qnulllrtsort = (1e-15 + sp.arange(self.mixture*self.isortlrt.shape[0]))/(self.mixture*self.isortlrt.shape[0])   
            # qnulllrtsort contains the expected distribution under the null hypothesis, 
            # which is a uniform distribution on (⁠\pi, 1). Usually the expected distribution
            # of the p-values in gwas is an uniform distribution on (0, 1).
            self.lrtsort = self.lrt[self.isortlrt]      
        
        # Here we define the bounds of the weights parameters, namely that they have to be semi-positive
        # and they cannot be higher than the 1 - a, where a is the weight of the 0 component
        bound = (0, self.mixture)
        bounds = [bound, bound]
        
        
        # Here we are going to define the constraint on our chi2 mixture's parameters
        # namely, the sum of the weights of the different chi2 has to sum up to 1, up
        # to a tolerance. The first weight, i.e. the one for the 0 dof component, is
        # left out of the fitting
        tolerance = 1e-6
        lowerlimit = self.mixture# - tolerance
        upperlimit = self.mixture# + tolerance
        
        linear_constraint = LinearConstraint([[1, 1]],  [lowerlimit], [upperlimit])
        
        x0 = 2*[self.mixture/2]
        res = minimize(self.fun, x0, method = 'trust-constr', constraints = [linear_constraint], 
                         options = {'verbose': 1}, bounds = bounds)
        
        print(res)
        return res.x, res.fun


        
    def fun(self, x):
        '''
        function where it is defined the error function to be minimized,
        that is the difference between:
        - the logarithm (it doesn't matter which base) of the expected p-values, namely 
          the uniform (pi_est, 1), where pi_est is the mixture parameter estimated from 
          the data, as detail at the beginning of this script
        '''

        a = x[0]
        b = x[1]
        # c = x[2]
        # d = x[3]
        
        base = sp.exp(1) # fitted params are invariant to this logarithm base (i.e. 10 or e)
        nfalse = (len(self.alteqnull) - sp.sum(self.alteqnull)) # number of statistics where the statistics is not 0
                                                                # which is the number of statistics that contribute to
                                                                # the Chi2 with dof = 1; the statistics equal to 0, 
                                                                # instead, contribute to the Chi2 with dof = 0 component
                                                                # of the Chi2 mixture.

        imax = int(sp.ceil(self.qmax*nfalse))  # of only non zero dof component
        
        # Obtaining the p-values from the statistics under the hypothesized null distribution
        p = a*st.chi2.sf(self.lrtsort[:imax], 1) + b*st.chi2.sf(self.lrtsort[:imax], 2)

        logp = sp.logn(base, p) # taking the logarithm of the above calculated pvals
        # Calculating the residuals, i.e. calculating the logarithm of the expected distribution of the p-values
        r = sp.logn(base, self.qnulllrtsort[0:imax]) - logp
        # Calculating the error 
        if self.abserr: # absolute error
            err = sp.absolute(r).sum()            
        else: # mean square error
            err = (r*r).mean()     
                     
        return err


    def sf(self, lrt, alteqnull, params):
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
        # the Chi2 with dof = 1, 2, 3 and 4
        pv = params[0]*st.chi2.sf(lrt, 1) + params[1]*st.chi2.sf(lrt, 2) #+ \
             #params[2]*st.chi2.sf(lrt, 3) + params[3]*st.chi2.sf(lrt, 4) 
        # the Chi2 with dof = 0, only for the statistics being 0                       
        pv[sp.array(alteqnull)] = 1
        return pv

    