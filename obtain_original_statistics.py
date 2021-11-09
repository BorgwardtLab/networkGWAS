#!/usr/bin/env python
# Author: Giulia Muzio

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
	genotype, phenotype, file_gene, fileout, file_sets = args
	
	genes = load_file(file_gene)
	
	# Running the method
	results_df = snp_set(test_snps = genotype, G0 = None, set_list = file_sets,
	                     pheno = phenotype, test = 'lrt', nperm = 0)

	statistics = process_results(results_df, genes)
	save_file(fileout, statistics)
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
	pvals:        gene-statistics numpy array
	'''
	sets = results_df['SetId']
	list_num = []
	for element in sets:
		list_num.append(int(re.findall(r'\d+', element)[0]))


	idx = np.argsort(list_num)
	statistics = (2*(results_df['LogLikeAlt'] - results_df['LogLikeNull'])).values
	statistics = statistics[idx]
	return np.c_[genes, statistics]


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
	parser.add_argument('--genotype',  required = False, default = 'data/plink/snp_matrix')
	parser.add_argument('--phenotype', required = False, default = 'data/plink/phenotype.pheno')
	parser.add_argument('--file_gene', required = False, default = 'data/gene_name.pkl')
	parser.add_argument('--fileout',  required = False, default = 'output/original_statistics.pkl')
	parser.add_argument('--file_sets',required = False, default = 'data/plink/neighborhood_list.txt')
	
	args = parser.parse_args()
	genotype   = args.genotype 
	phenotype  = args.phenotype 
	file_gene  = args.file_gene
	fileout    = args.fileout
	file_sets  = args.file_sets
	return genotype, phenotype, file_gene, fileout, file_sets


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
