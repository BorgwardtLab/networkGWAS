from __future__ import absolute_import
import numpy as np
import time
import fastlmm.util.standardizer as stdizer


def standardize(snps, blocksize = None, standardizer = stdizer.Unit(), force_python_only = False): # USED!
    '''
    Function for standardizing the SNPs.  
    Does in-place standardization.       
    '''
    if isinstance(standardizer, str):
        standardizer = standardizer.factor(standardizer)

    if blocksize is not None and blocksize >= snps.shape[1]: # If blocksize is larger than the # of snps, set it to None
        blocksize = None

    return standardizer.standardize(snps, blocksize = blocksize, force_python_only = force_python_only)