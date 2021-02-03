'''
Function for performing fastlmm snp-set on lists
of SNPs belonging to 1-degree neighbourhoods
constructed starting from each of the permuted
gene-snp mapping.
Through this script it is possible to obtain
the p-values for the permuted network, which then 
will be adjusted using the degree-preserving 
permutation strategy.

NOTE: this script performs the computation of
	  the j-th permutation

Inputs:
- 'data/plink/snp_matrix'                 bem/bim/fim files of the SNPs matrix where there are
										  the SNPs to test, e.g. the SNPs in the neighborhood
										  to test the association with the phenotype

- 'data/plink/phenotype.pheno'            file with the phenotype in plink format

- 'output/permutations/ \
   nb_files/list_nb1_' + j + '.txt':      input file for FaST-LMM-Set function. It is a .txt
										  file having on one column the name of the SNPs, and
										  on the other column the name of the set they belong to. 
										  This has the information about the SNPs obtained by 
										  performing 1-degree aggregation of the SNPs, e.g. each
										  neighborhood is represented by a set of SNPs on the j-th 
										  permuted network

- 'data/gene_name.pkl':   				  numpy vector containing the names of the genes 
										  included in the network

- 'output/permutations/pvalues/ \
   pvalues_' + str(j) + '.pkl':           path where to save the pvalues obtained on the j-th 
   								          permuted network

Command-line arguments:
--test          						  kind of statistical test to be used to obtain the 
										  association score with FaST-LMM-Set function. Could 
										  be either 'lrt' or 'sc-davies'. Default is "lrt". It 
										  should be consistent with what was chosen for the original
										  p-values.

--j                                       index of the permutation for which to launch the FaST-LMM
										  snp-set function. This is done for making it easy to paralle-
										  lise the computation.
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
	test, j = args
	
	# LOADING DATA
	# 1. SNP matrix's location in plink format
	genotype = 'data/plink/snp_matrix' # it should be in bed/bim/fam format
	# 2. Phenotype's plink file location
	phenotype = 'data/plink/phenotype.pheno'
	# 3. File with the set of SNPs corresponding to the 1-degree neighbourhoods
	file_sets = 'output/permutations/nb_files/list_nb1_' + j + '.txt'
	# 4. Loading the names of the genes
	genes = load_file('data/gene_name.pkl')
	# Output dir
	outdir = 'output/permutations/pvalues/'
	if(not os.path.exists(outdir)):
			os.makedirs(outdir) 
	
	# Running the method
	results_df = snp_set(test_snps = genotype, G0 = None, set_list = file_sets,
	                     pheno = phenotype, test = test, nperm = 0)

	pvals = process_results(results_df, genes)
	save_file(outdir + 'pvalues_' + j + '.pkl', pvals)
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
				obtaining p-values; lrt or sc-davies
	j:          index of permutation
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', required = False, default = "lrt")
	parser.add_argument('--j', required = True)
	args = parser.parse_args()

	test = args.test
	j = args.j
	return test, j


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)

	
