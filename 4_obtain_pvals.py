#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script to compute the p-values.
Having obtained the loglikelihood ratio (lrt) statistics 
on the non-permuted and the permuted settings with the
script 3_LMM.py, this current script performs:

1) pooling of the lrt statistics obtained on the permuted
settings to obtain the distribution of the statistics under
the null hypothesis

2) calculation of the p-values from the lrt on the original 
setting and the null distribution
'''

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import *


def main(args):

	if(not os.path.exists(args.dirnd)):
		os.makedirs(args.dirnd)

	if(not os.path.exists(args.dirpv)):
		os.makedirs(args.dirpv)

	# Loading observed statistics
	observed_lrt = load_file(args.inpath)
	
	# Null distribution construction
	null_distr  = pooling_null_distr(args.inpathperm, args.nperm)
	# Genomic inflation factor
	lambda_gc   = lambda_gc_calc(observed_lrt, null_distr)
	# p-values calculation
	pvals       = pval_computation(null_distr, observed_lrt)
	
	# null distribution and qq-plot 
	null_distr_histogram(null_distr, args)
	qqplot(pvals, lambda_gc, args)

	# saving null distribution and p-values 
	save_file('{}/{}'.format(args.dirnd, args.outpathnd), null_distr)
	save_file('{}/{}'.format(args.dirpv,args.outpathpv), [np.array(list(observed_lrt.keys())), pvals])


def null_distr_histogram(null_distr, args):
	'''
	Function for obtaining the histogram of the 
	statistics under the null distribution.

	Input
	-----------
	null_distr:  null distribution of the test
				 statistics
	args:        command-line arguments

	Output
	-----------
	'''
	plt.figure()
	plt.hist(null_distr, bins = 100, density = True)
	plt.xlabel('log-likelihood ratio')
	plt.ylabel('density')
	plt.savefig('{}/{}'.format(args.dirnd, args.fignd))


def lambda_gc_calc(observed, null_distr):
	'''
	Function to calculate the genomic inflation factor, i.e.,
	the ratio between the median of the observed statistics
	and the statistics under the null distribution.

	Input
	---------------- 
	observed:    observed statistics, e.g., lrt statistics
				 on the non-permuted scenario
	null_distr:  null distribution of the test statistics

	Output
	---------------
	lambda_gc:   genomic inflation factor
	'''
	observed_median = np.median(list(observed.values()))
	expected_median = np.median(null_distr)
	lambda_gc = observed_median / expected_median
	print('genomic inflation factor: {}'.format(lambda_gc))
	return np.round(lambda_gc, 3)


def pooling_null_distr(perm, nperm):
	'''
	Function for obtaining the null distribution of the test
	statistics under the null hypothesis. This is done by
	pooling the lrt statistics obtained on the permuted settings

	Input
	------------
	perm:		 base-name to the statistics obtained per each
				 different permutation
	nperm:       total number of permutations
	Output
	-----------
	flat_list:   null distribution of the test statistics
	'''
	pool = []
	for i in range(nperm):
		perm_i = load_file(perm + str(i) + '.pkl')
		pool.append(list(perm_i.values()))
	
	flat_list = [item for sublist in pool for item in sublist]
	return flat_list


def qqplot(pvals, lambdagc, args, h1=None, figsize=[5,5]):
	'''
	performs a P-value QQ-plot in -log10(P-value) space
	
	Input
	--------------
	pvals:    p-values
	lambdagc: genomic inflation factor
	args:     command-line arguments

	Output
	--------------
	'''    
	plt.figure(figsize=figsize) 

	maxval = 0
	M = pvals.shape[0]
	pnull = np.arange(1, M + 1)/M # uniform distribution for the pvals
	# Taking the log10 of expected and observed
	qnull = -np.log10(pnull)            
	qemp  = -np.log10(np.sort(pvals))

	# Taking medians and plotting it
	qnull_median = np.median(qnull)
	qemp_median = np.median(qemp)

	xl = r'$-log_{10}(P)$ observed'
	yl = r'$-log_{10}(P)$ expected'
	if qnull.max() > maxval:
	    maxval = qnull.max()                
	plt.plot(qnull, qemp, 'o', markersize = 2)
	plt.plot([0, qnull.max()], [0, qnull.max()],'k')
	plt.ylabel(xl)
	plt.xlabel(yl)

	props = dict(boxstyle='round', facecolor='azure', alpha=0.5)

	plt.text(qnull.max() - 0.7, 0.05, r'$\lambda_{GC} = $' + str(lambdagc), 
	    			fontsize=10, verticalalignment='baseline',  bbox=props)
	plt.hlines(y = qemp_median, xmin = -0.15, xmax=qnull_median, 
			linestyle = '--', color = 'silver', linewidth = 0.8)
	plt.vlines(x = qnull_median, ymin = -0.15, ymax = qemp_median,
			linestyle = '--', color = 'silver', linewidth = 0.8)
	plt.axis(xmin = -0.15, ymin=-0.15)
	plt.plot(qnull_median, qemp_median, 'o', markersize = 2, color='silver')
	
	plt.savefig('{}/{}'.format(args.dirpv ,args.figpv), dpi = 300)



def pval_computation(null_distr, observed_dict):
	'''
	Function for the calculation of the p-values 
	given the lrt statistics on the original 
	(non-permuted) setting and the distribution
	of the lrt statistics under the null hypothesis

	Input
	--------------
	null_distr:   null distribution of the lrt 
				  statistics
	observed:     lrt statistics on the non-permuted
				  setting

	Output
	--------------
	pvals:        pvalues
	'''
	n_values = len(null_distr)
	observed = list(observed_dict.values()) 

	pvals = []
	for o in observed:
		pval = ((null_distr >= o).sum() + 1)/(n_values + 1)
		pvals.append(pval)

	return np.array(pvals)


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	-----------
	
	Output
	-----------
	args.inpath:	   path to the lrt statistics on the non-permuted
					   setting
	args.inpathperm:   base-name of the llr statistics obtained on the
					   permutated settings, e.g., 'results/llr/permuted/llr_',
	args.nperm:		   total number of permutations
	args.dirnd:		   folder where to save the null distribution
	args.dirpv:        folder where to save the p-values
	args.fignd:		   path to the figure of the histogram of the null
					   distributions
	args.figpv:        path to the figure of the qqplot of the p-values
	args.outpathnd:    path to the null distribution file
	args.outpathpv:    path to the p-values file
	
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--inpath',     required = False, default = 'results/llr/llr.pkl',
						help = 'path to the lrt statistics on the non-permuted setting.')
	parser.add_argument('--inpathperm', required = False, default = 'results/llr/permuted/llr_',
						help = 'base-name of the lrt statistics on the permuted settings.')
	parser.add_argument('--nperm',      required = False, default = 300, type = int, 
						help = 'total number of permutations')
	parser.add_argument('--dirnd',      required = False, default = 'results/null_distr/',
						help = 'folder where to save the null distribution')
	parser.add_argument('--dirpv',      required = False, default = 'results/pvals/',
						help = 'folder where to save the pvals')
	parser.add_argument('--fignd',      required = False, default = 'null_distr.png',
						help = 'name of the figure of the null distribution')
	parser.add_argument('--figpv',      required = False, default = 'qqplot.png' ,
						help = 'name of the figure of the qqplots of the p-values')
	parser.add_argument('--outpathnd',  required = False, default = 'null_distr.pkl',
						help = 'name of the null distribution file')
	parser.add_argument('--outpathpv',  required = False, default = 'pvals.pkl',
						help = 'name of the p-values file')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
	