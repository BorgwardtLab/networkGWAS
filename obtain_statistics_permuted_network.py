#!/usr/bin/env python
# Author: Giulia Muzio

'''
Function for performing fastlmm snp-set on lists
of SNPs belonging to 1-degree neighbourhoods
constructed starting from each of the permuted
gene-snp mapping.
Through this script it is possible to obtain
the p-values for the permuted network, which then 
will be adjusted using the SNPs-block-preserving 
permutation strategy.

NOTE: this script performs the computation of
	  the j-th permutation

Inputs:
- 'data/plink/snp_matrix'     			  bem/bim/fim files of the SNPs matrix where there are
													  the SNPs to test, e.g. the SNPs in the neighborhood
													  to test the association with the phenotype

- 'data/plink/phenotype.pheno'            file with the phenotype in plink format

- 'output/permutations/neighborhoods_' +
   blocksize + /list_nb1_' + j + '.txt':  input file for FaST-LMM-Set function. It is a .txt
										  				file having on one column the name of the SNPs, and
													   on the other column the name of the set they belong to. 
										  				This has the information about the SNPs obtained by 
										  				performing 1-degree aggregation of the SNPs, e.g. each
										  				neighborhood is represented by a set of SNPs on the j-th 
										  				permuted network

- 'output/permutations/statistics_' +
 	blocksize + 'statistics_' + str(j) 
 	+ '.pkl':          						   path where to save the pvalues obtained on the j-th 
   								          		permuted network

Command-line arguments:
--test          						 			kind of statistical test to be used to obtain the 
										  				association score with FaST-LMM-Set function. Could 
										  				be either 'lrt' or 'sc-davies'. Default is "lrt". It 
										  				should be consistent with what was chosen for the original
										  				p-values.
--j                                       index of the permutation for which to launch the FaST-LMM
										  				snp-set function. This is done for making it easy to paralle-
										  				lise the computation.
--blocksize:										integer; it's the size of the blocks.
--genotype:											PLINK format (bed/bim/bam) file for the genotype
--phenotype:										.pheno format for the phenotype
--outdir:											string; name of the main output folder
'''

import numpy as np
import pandas as pd
import argparse
from fastlmm.association import snp_set
import fastlmm
from utils import *
import re
import os


def main(args):
	# command line arguments
	test, j, blocksize, genotype, phenotype, outdir = args
	
	# File with the set of SNPs corresponding to the 1-degree neighbourhoods
	file_sets =  outdir + 'neighborhoods_' +  blocksize +  '/list_nb1_' + j + '.txt'
	
	# Output dir for the statistics on permuted settings
	outdir = outdir + 'statistics_' + blocksize +'/'
	if(not os.path.exists(outdir)):
			os.makedirs(outdir) 
	
	# Running the method
	results_df = snp_set(test_snps = genotype,  G0 = None, set_list = file_sets,
	                         pheno = phenotype,   test = test,    nperm = 0)

	statistics = process_results(results_df)
	save_file(outdir + 'statistics_' + j + '.pkl', statistics)
	return 0


def process_results(results_df):
	'''
	Function for obtaining the pvalues in the correct order

	Input
	-------------
	results_df:   results from fastlmm snp-set function
	genes:        name of the genes 
	
	Output
	-------------
	pvals:        gene-pvalue numpy array
	'''
	statistics = (2*(results_df['LogLikeAlt'] - results_df['LogLikeNull'])).values
	return statistics


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------

	Output
	---------
	test:       kind of statistical test to be performed for
				obtaining p-values; lrt or sc-davies
	j:          index of permutation
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--test',      required = False, default = "lrt")
	parser.add_argument('--j',         required = True)
	parser.add_argument('--blocksize', required = False, default = '50')
	parser.add_argument('--genotype',  required = False, default = 'data/plink/snp_matrix')
	parser.add_argument('--phenotype', required = False, default = 'data/plink/phenotype.pheno')
	parser.add_argument('--outdir',    required = False, default = 'output/permutations/')
	
	args = parser.parse_args()

	test       = args.test
	j          = args.j
	blocksize  = args.blocksize
	genotype   = args.genotype 
	phenotype  = args.phenotype 
	outdir     = args.outdir
	return test, j, blocksize, genotype, phenotype, outdir


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)

	
