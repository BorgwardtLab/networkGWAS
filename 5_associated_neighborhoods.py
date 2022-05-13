#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script to perform benjamini-hochberg procedure
to correct for multiple testing and control
the false discovert rate (FDR) at a defined 
level.
'''

import os
import sys
import argparse

import numpy as np

from utils import *
from IPython import embed

def main(args):
	# Loading the causal genes
	nw = load_file(args.nw)
	
	if(len(args.pv) == 1):
		# Benjamini-hochberg procedure
		pv = load_file(args.pv[0])
		pos = benjamini_hochberg(pv[1], pv[0], args.q1)
		
	else:
		# Hierarchical benjamini-hochberg-based procedure
		pv, cg = read_pv(args.pv, args.cg)
		pos_pheno = step1(pv, args)
		for pheno in pos_pheno: # step 2
			print('Phenotype {}:'.format(pheno))
			pos = benjamini_hochberg(pv[pheno][1], pv[pheno][0], args.q2*len(pos_pheno)/len(pv))
			



def step1(pvs, args):
	'''
	Step 1 for the hierarchical multiple testing correction
	procedure. 

	Input
	------------
	pvs:		 dictionary of the p-values for the different
				 phenotypes
	args:        command-line arguments

	Output
	------------
	pos_pheno:   phenotypes where the null hypothesis is
				 rejected, namely where the hypothesis of
				 no genetic signal is rejected
	'''
	phenos = np.array(list(pvs.keys()))
	p_t = []
	for p in pvs:
		pv = pvs[p][1]
		p_t.append(simes_pval(pv))

	pos_pheno = benjamini_hochberg(np.array(p_t), phenos, args.q1)
	print('Phenotypes where the hypothesis is rejected:')
	[print(ph) for ph in pos_pheno]
	print('-----------------------')
	return pos_pheno


def simes_pval(pv):
	'''
	Function for calculating the Simes' p-value
	from a "family" of p-values.

	Input
	---------
	pv: 	  p-values from a family

	Output
	---------
	simespv:  simes p-value 
	'''
	pvals = np.sort(pv)
	ranks = np.arange(1, len(pvals) + 1)
	multiplied = pvals*len(ranks)
	results = multiplied/ranks
	simespv = np.min(results)
	return simespv



def benjamini_hochberg(pval_not_sorted, test_not_sorted, q):
	'''
	Function for performing the benjamini hochberg (B-H) 
	procedure

	Input
	--------------
	pval_not_sorted:	pvalues, not necessary sorted.
	test_not_sorted:	tests ID, ordered according to 
						pval_not_sorted
	q:					the FDR level at which we are 
						controlling using the B-H procedure

	Output
	--------------
	pos:				test ID for which the null hypothesis is 
						rejected 
	'''
	index  = np.argsort(pval_not_sorted)
	pvals  = pval_not_sorted[index]
	test   = test_not_sorted[index]
	n_test = len(pvals)
	
	thresholds = (np.arange(1, n_test + 1)*q)/n_test
	v_set = np.where(pvals <= thresholds)[0]
	predicted_pos = np.zeros(n_test).astype(bool)
	if(len(v_set) > 0):
		V = np.max(v_set)
		predicted_pos[:(V + 1)] = True

	pos = test[predicted_pos]
	return pos



def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	----------

	Output
	----------
	args.cg:   path to the array of causal genes
	args.nw:   path to the network
	args.pv:   path to the pvalue. Note that this command-line argument
			   can accept multiple p-values' paths
	args.q1:   FDR level we want to control at using B-H
	args.q2:   FDR level we want to control at using B-H-based hierarchical
			   procedure
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--nw', required = False, default = 'data/PPI_adj.pkl')
	parser.add_argument('--pv', required = False, nargs = '+',
						default = ['results/pvals/pvals.pkl'])
	parser.add_argument('--q1', required = False, type = float, default = 0.1)
	parser.add_argument('--q2', required = False, type = float, default = 0.1)
	args = parser.parse_args()

	return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
	