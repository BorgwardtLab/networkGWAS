'''
Obtaining the null distribution from the p-values
obtained by appling snp-set function on the permuted
network. The null distribution is obtained by merging
all the p-values from all the permutations.
Afterwards, we obtain the adjusted p-values
by calculating the fraction of p-values in the null
distribution which are less or equal than the original
pvalues, i.e. the ones obtained on the real network. 
Then, we just use Benjamini-Hochberg for obtaining the
list of neighborhoods predicted as associated with
the phenotype. 

Inputs:
- 'data/gene_name.pkl':            numpy vector containing the names of the genes 
								   included in the network

- 'output/original_pvalues.pkl':   path to the file where the original p-values are saved,
								   i.e. the pvalues obtained on the original network. These
								   p-values has to be adjusted using the degree-preserving
								   permutation strategy

- 'output/permutations/pvalues/ \
   pvalues_' + str(j) + '.pkl':    path to where the pvalues obtained on the j-th permuted network
   								   are saved  

- 'output/adj_pval.pkl':           path where to save the adjusted p-values obtained using the
								   degree-preserving permutation strategy

- 'data/causal_genes.pkl':         name of the causal genes


Command-line arguments:
--j                                index of the permutation for which to launch the FaST-LMM
								   snp-set function. This is done for making it easy to paralle-
								   lise the computation.
'''

import numpy as np
import pandas as pd
from IPython import embed
import argparse
from utils import *
import re


def main(args):
	nperm = args
	# Setting parameters
	FDR = 0.1

	# Loading files
	gene_name = load_file('data/gene_name.pkl') # genes
	original_pvals = load_file('output/original_pvalues.pkl')
	filename = 'output/permutations/pvalues/pvalues_'
	g = len(gene_name) # obtaining the number of genes, which is the num of tests
	causal = load_file('data/causal_genes.pkl')

	# 1. Obtaining null distribution
	null_distr_fdr = obtain_null_distributions(nperm, g, filename)
	# 2. Obtaining adjusted p-values
	adj_pval = pvalComputation(original_pvals, null_distr_fdr, nperm*g)
	save_file('output/adj_pval.pkl', adj_pval)

	# 3. Applying Benjamini-Hochberg method for predicting the associated
	#    neighbourhoods
	pred_pos = benjamini_hochberg(adj_pval, original_pvals[:, 0], g, FDR)
	
	# 4. Contingency table, precision, recall
	performance(pred_pos, causal, g)
	return 0


def benjamini_hochberg(pval, genes, n_test, q):
	'''
	benjamini hochberg procedure

	Input
	--------------------------
	pval:      p-values
	genes:     name of the genes
	n_test:    number of tests
	q:         FDR threshold

	Output
	--------------------------
	pos_gene:  name of genes predicted as associated
	'''
	idx = np.argsort(pval)
	order_pval = pval[idx]
	order_gene = genes[idx] 
	thresholds = (np.arange(1, n_test + 1)*q)/n_test
	v_set = np.where(order_pval <= thresholds)
	predicted_pos = np.zeros(len(pval)).astype(int)
	if(len(v_set[0]) > 0):
		V = np.max(v_set)
		predicted_pos[:(V + 1)] = 1

	pos_gene = order_gene[predicted_pos.astype(bool)]
	return pos_gene


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


def obtain_null_distributions(nperm, g, filename):
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
		pvalues = load_file(filename + str(j) + '.pkl')
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
	
