'''
Function for performing fastlmm snp-set on lists
of SNPs belonging to 1-degree neighbourhoods
constructed starting from each gene.
Through this script it is possible to obtain
the original p-values, which then will be adjusted
using the degree-preserving permutation strategy.

Inputs:
- 'data/plink/snp_matrix'                 bem/bim/fim files of the SNPs matrix where there are
										  the SNPs to test, e.g. the SNPs in the neighborhood
										  to test the association with the phenotype

- 'data/plink/phenotype.pheno'            file with the phenotype in plink format

- 'data/plink/neighborhood_list.txt':     input file for FaST-LMM-Set function. It is a .txt
										  file having on one column the name of the SNPs, and
										  on the other column the name of the set they belong to. 
										  This has the information about the SNPs obtained by 
										  performing 1-degree aggregation of the SNPs, e.g. each
										  neighborhood is represented by a set of SNPs

- 'data/gene_name.pkl':   				  numpy vector containing the names of the genes 
										  included in the network

- 'output/original_pvalues.pkl':          path to the file where to save the original p-values,
										  i.e. the pvalues obtained on the original network. These
										  p-values has to be adjusted using the degree-preserving
										  permutation strategy.

Command-line arguments:
--test          						  kind of statistical test to be used to obtain the 
										  association score with FaST-LMM-Set function. Could 
										  be either 'lrt' or 'sc-davies'. Default is "lrt".
'''

import numpy as np
import pandas as pd
import argparse
from fastlmm.association import snp_set
import fastlmm
from utils import *
import re


def main(args):
	# command line arguments
	test = args
	
	# LOADING DATA
	# 1. SNP matrix's location in plink format
	genotype = 'data/plink/snp_matrix' # it should be in bed/bim/fam format
	# 2. Phenotype's plink file location
	phenotype = 'data/plink/phenotype.pheno'
	# 3. File with the set of SNPs corresponding to the 1-degree neighbourhoods
	file_sets = 'data/plink/neighborhood_list.txt'
	# 4. Loading the names of the genes
	genes = load_file('data/gene_name.pkl')
	# Defining the output filename
	fileout = 'output/original_pvalues.pkl'
	
	# Running the method
	results_df = snp_set(test_snps = genotype, G0 = None, set_list = file_sets,
	                     pheno = phenotype, test = test, nperm = 0)

	pvals = process_results(results_df, genes)
	save_file(fileout, pvals)
	return 0


def process_results(results_df, genes):
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
	sets = results_df['SetId']
	list_num = []
	for element in sets:
		list_num.append(int(re.findall(r'\d+', element)[0]))


	idx = np.argsort(list_num)
	pvals_not_ordered = results_df['P-value']
	pvals = pvals_not_ordered[idx]
	return np.c_[genes, pvals]


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------

	Output
	---------
	test:       kind of statistical test to be performed for
				obtaining p-values
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', required = False, default = "lrt")
	args = parser.parse_args()

	test     = args.test
	return test


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)

	
