'''
Compared to the original implementation at
https://github.com/fastlmm/FaST-LMM/
this file has been modified by Giulia Muzio
'''
import time
import argparse
import pandas as pd

from association import FastLmmSet
from association.FastLmmSet import FastLmmSet
from pysnptools.util.mapreduce1.runner import *


def networkGWAS_snpSet(test_snps, set_list, phenotype, covariate = None, 
                output_file = None, test_type = "lrt", kernel = None, 
                standardize_data = None):
    """
    Function performing GWAS on sets of snps

    Input
    ----------------------------------------------------------------------------------------------
    test_snps:      The base name of the file containing the SNPs for alternative kernel. 
                    The file must be in PLINK Bed format (string)
    set_list:       The name of a tab-delimited file defining the sets. The file should 
                    contain two-columns 'snp' and 'set' (string)
    pheno:          The name of a file containing the phenotype. The file must be in PLINK 
                    phenotype format (string)
    covariate:      covariate information, optional: The name of a file in PLINK phenotype 
                    format (string)
    output_file:    Name of file to write results to, optional. If not given, no output file 
                    will be created (string)
    test_type:      'lrt' (default) (string)

    Output
    --------------------------------------------------------------------------------------------------
    results:        Pandas dataframe with one row per set.

    """


    if(kernel == "lin"):
        print('LINEAR KERNEL')
        KERNEL = {'type':'linear'}
    elif(kernel == 'poly'):
        print('POLYNOMIAL KERNEL')
        KERNEL = {'type':'polynomial'}


    nullModel = {'effect':'fixed', 'link':'linear'}
    altModel  = {'effect':'mixed', 'link':'linear'}
    
    fastlmm_set = FastLmmSet(outfile = output_file, phenofile = phenotype, alt_snpreader = test_snps, 
        altset_list = set_list, covarfile = covariate, test = test_type, autoselect = False,
        nullModel = nullModel, altModel = altModel, kernel = KERNEL, standardize_data = standardize_data )


    # Running through all the sets and permutations
    sequence = fastlmm_set.work_sequence()

    # the following few lines of code permit to run
    # the actual analysis. In fact, sequence is a generator
    # which return a function. With next(generator), the
    # function is run.
    flag = True
    result_list = []
    while(flag):
        try:
            result = next(sequence)
            result_list.append(result[0])
        except StopIteration: 
            flag = False
        
    observed_statistics, null_distribution = fastlmm_set.reduce(result_list)
    return observed_statistics, null_distribution
    


