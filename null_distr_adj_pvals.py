# Obtaining the null distribution
# and the adjusted p-values

import numpy as np
import pandas as pd
from IPython import embed
import argparse
from utils import *
import re


def main(args):
	nperm = args

	# Loading files
	gene_name = load_file('data/gene_name.pkl') # genes
	original_pvals = load_file('output/original_pvalues.pkl')
	g = len(gene_name)

	null_distr_fdr = obtain_null_distributions(nperm, g)
	adj_pval = pvalComputation(original_pvals, null_distr_fdr, nperm)
	save_file('output/adj_pval.pkl', adj_pval)
	return 0



def pvalComputation(res, null_distr, nperm):
	'''
	Cmputation of the p-values.
	It's calculated as the proportion of 
	p-values on the total number of p-values 
	which are no high than the original p-value

	Input
	--------------
	res:    	  genes names (the center node of the 
				  neighborhood and their original p-vals)
	null_distr:   null distribution
	nperm:        number of permutations

	Output
	--------------
	adj_pval:     adjusted pvalues

	'''
	adj_pval = []
	for pval in res[:, 1]:
		num = sum(np.array(null_distr) <= float(pval))
		adj_pval.append(num/nperm)


	return np.array(adj_pval) 


def obtain_null_distributions(nperm, g):
	'''
	Function for obtaining the null distribution
	as the union of all the pvalues. 

	Input
	-------------
	nperm:            number of permutations
	g:                number of genes
	
	Output
	-------------
	null_distrFDR:    null distribution
	'''
	
	null_distrFDR = np.array([])
	
	for j in range(nperm):
		pvalues = load_file('output/permutations/pvalues/pvalues_' + str(j) + '.pkl')
		null_distrFDR = np.concatenate((null_distrFDR, pvalues[:, 1].astype(float)))
	

	return null_distrFDR


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------
	Output
	---------
	nperm:	number of permutations
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--nperm', required = False, default = 1000, type = int)
	args = parser.parse_args()

	nperm = args.nperm
	return nperm


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
	
