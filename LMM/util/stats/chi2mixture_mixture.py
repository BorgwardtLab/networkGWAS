
'''
Adapted code from fastlmm implementation.
This is the script for fitting the mixture
parameter only. In this script, it is obtained
by just calculating the proportion of statistics
in the permutations that are zero; the threshold 
we choose is 1e-10.
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


class chi2mixture_mixture(object):
    '''
    mixture here denotes the weight on the non-zero dof compnent
    '''

    def __init__(self, lrt, tol = 0.0, alteqnull = None, dof = None):
        '''
        Input:
        lrt             [Ntests] vector of test statistics
        a2 (optional)   [Ntests] vector of model variance parameters
        top (0.0)       tolerance for matching zero variance parameters or lrts
        qmax (None)      only the top qmax quantile is used for the fit
        '''
        self.lrt = lrt # statistics from the permutations       
        self.alteqnull = alteqnull # index where the lrt = 0
        self.dof = dof # degree of freedom
        self.mixture = None
        self.tol = tol # tolerance for the fitting
        self.__fit_mixture()
        self.isortlrt = None

    def __fit_mixture(self):
        '''
        fit the mixture component
        '''
        if self.tol < 0.0:
            logging.info('tol has to be larger or equal than zero.')
        if self.alteqnull is None:
            self.alteqnull = self.lrt <= 1e-10
            return self.alteqnull, self.mixture


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
        mixture = 1 - (sp.array(self.alteqnull).sum())/(sp.array(self.alteqnull).shape[0])
        print('Fitted mixture:' + mixture)
        lrt =  lrt.astype(float)
        # the Chi2 with dof = 1
        pv = mixture*st.chi2.sf(lrt, self.dof)
        # the Chi2 with dof = 0, only for the statistics being 0                       
        pv[sp.array(alteqnull)] = 1
        return pv

    