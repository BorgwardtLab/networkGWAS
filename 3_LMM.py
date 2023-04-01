#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for performing the maximum restricted likelihood
to calculate the likelihood ratio test for the neighborhoods
by employing the FaST-LMM-set implemetation obtainable at
https://github.com/fastlmm/FaST-LMM/

This script can be used on both the original (i.e., non
permuted scenario) and the multiple permuted scenarios. 

This script can easily be run in parallel for different
permutations as j is a command line argument.
'''

import os
import sys
import argparse

import numpy as np

from utils import *
sys.path.insert(0, 'LMM/')
from LMM.association.networkGWAS_snpSet import networkGWAS_snpSet


def main(args):

	# Input files
	genotype  = args.genotype # bed bim fam file 
	phenotype = args.phenotype
	
	if(not os.path.exists(args.odir)):
		os.makedirs(args.odir)
	
	if(not args.j):
		snp_set_list = args.nbs
		output       = '{}/{}'.format(args.odir, args.ofile)

		# running fastlmm snp-set
		observed_statistics, null_distribution = \
		networkGWAS_snpSet(test_snps = genotype, set_list = snp_set_list, 
							phenotype = phenotype, test_type = "lrt",
							covariate = args.covariate, kernel = args.kernel,
							standardize_data = True)

		save_file(output, observed_statistics)
	else:
		if(len(args.j) > 2): raise NameError('Wrong permutation indexes!')
		
		for j in range(args.j[0], args.j[-1] + 1):
			snp_set_list = args.nbs + str(j) + '.txt'
			output       = '{}/{}{}.pkl'.format(args.odir, args.ofile, j)

			# running fastlmm snp-set
			observed_statistics, null_distribution = \
			networkGWAS_snpSet(test_snps = genotype, set_list = snp_set_list, 
								phenotype = phenotype, test_type = "lrt",
								covariate = args.covariate, kernel = args.kernel,
								standardize_data = True)

			save_file(output, observed_statistics)
		 


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	-----------

	Output
	-----------
	args.genotype:   path to the bed/bim/fam genotype
	args.phenotype:  path to the phenotype (in plink format)
	args.nbs:        path to the file where the sets to tests, e.g.,
				     the neighborhoods, are defined
	args.covariate:  path to the (optional) file where the covariates
				     are saved
	args.kernel:     string, either "lin" or "poly", that defines which 
				     kernel to use for measuring the similarities between
				     the SNPs in the test-set, i.e., the neighborhood
	args.j:			 permutation number. Optional. If present, then the 
				     likelihood ratio for the neighborhoods is obtained 
				     on the j-th permuted setting. Otherwise, on the non
				     permuted setting (which should be done once)
	args.odir:		 output folder. For non-permuted setting, it could
					 be: "results/llr/"

					 For the permuted settings, it could be:

					 "results/llr/permuted/" 
	args.file:		 name of the output file. For example, "llr.pkl" in
					 case of the non-permuted setting, and a base-name "llr_"
					 in case of the permuted settings, since the script will 
					 add the permutation index and the file format.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--genotype',  required = False, default = 'data/genotype')
	parser.add_argument('--phenotype', required = False, default = 'data/y_50.pheno')
	parser.add_argument('--nbs',       required = True, help = 'neighborhoods file. Under\
							   the current folder structure, it would be "results/settings/neighborhoods.txt"\
							   in case of the non-permuted scenario, and \
							   "results/settings/permutations/neighborhoods/nbs_" otherwise.')
	parser.add_argument('--covariate', required = False)
	parser.add_argument('--kernel',    required = True, choices = ['lin', 'poly'],
								help = 'type of the kernel for modeling \
								the similarities between the SNPs in the\
								neighborhoods.')
	parser.add_argument('--j',  nargs='+',   type=int,     required = False, 
								help = 'permutation index. If not given,\
							   the analysis is performed on the non-permuted setting.')
	parser.add_argument('--odir',       required = True, 
							   help = 'folder where to save the results. For example,\
							    "results/llr/"" in case of the non-permuted scenario,\
							    and "results/llr/permuted/" otherwise.')
	parser.add_argument('--ofile',      required = True, 
							   help = 'filename for the results. "llr.pkl" in case\
							   of the non-permuted scenario, and "llr_" otherwise.')
	args = parser.parse_args()
	return args
	


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
	