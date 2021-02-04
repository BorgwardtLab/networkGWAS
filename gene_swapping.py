#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for saving the permuted genes;
the permutation is done by following
the degree-permutation strategy

File in input:
- 'data/ppi.pkl': 		                  PPI network in form of pandas DataFrame. The
										  keys of the data frame are the names of the 
										  genes included in the network. The values of
										  the dataframe is the non-weighted adjacency 
										  matrix of the PPI network, e.g. 1 when there is
										  an edge between the 2 genes, and 0 when there is
										  no edge, e.g. no interaction

- 'data/gene_name.pkl':   				  numpy vector containing the names of the genes 
										  included in the network

- 'output/permutations/permuted_genes/':  string where to save the permuted 
										  genes.

Command-line arguments:
--nperm:    				              integer; it's the number of permutation to perform
										  minimum should be 1000.
'''

import numpy as np
import pandas as pd
import argparse
from utils import *
import re


def main(args):	
	nperm = args

	# LOADING INPUT FILES
	network = load_file('data/ppi.pkl') # network
	gene_name = load_file('data/gene_name.pkl') # genes
	outdir = 'output/permutations/permuted_genes/'
	
	# Calling the function for performing the 
	# degree-preserving permutation technique
	degree_permutations(network, gene_name, nperm, outdir)
	
	return 0
	

def degree_permutations(network_df, gene_name, nperm, minimum = 5):
	'''
	Function for performing the swapping of genes having the same 
	degree (or close); it directly save the permuted genes in the 
	output folder

	Input
	-------------------
	network_df:  pandas dataframe containing the network
				 in form of adjacency matrix
	gene_name:   genes
	nperm:       number of permutations
	mimimun:     minimum cardinality of genes in the same
				 swapping group
	
	Output
	-------------------
	'''
	# finding the degrees
	network = network_df.values
	degree = network.sum(axis = 0) # also axis = 1 is fine since it's simmetric
	deg, count = np.unique(degree, return_counts = True)
	
	# - the degrees having numerosity equal or higher than 5 are taken 
	#   separately
	# - the degrees with cardinality less than 5 are grouped with the
	#   closest degrees to reach at leat 5 as cardinality
	# we do that starting from the end, because usually the genes with 
	# higher degree have lower cardinality

	swapping_group = []
	temp, counts, flag = [], 0, False
	for d, c in zip(deg[::-1], count[::-1]):
		if((c < minimum) | (flag == True)): # if for the current degree there are
											# no more than minimum genes
			temp.append(d)
			flag = True
			counts += c 
			if (counts >= minimum):
				swapping_group.append(temp)
				flag = False
				counts = 0
				temp = []
			elif(d == deg[0]): # if the last neighbourhood hasn't been 
							   # assigned yet to a "degree group" which
							   # could reach the minimum number defined
				swapping_group[-1] = [swapping_group[-1], d]
		else: # if the number of genes having the current degree is enough
		      # e.g. higher than minimum
			swapping_group.append(d)

	swapping_group = swapping_group[::-1]
	
	# actual degree-preserving permutation technique
	for j in range(nperm):
		tot_perm = np.empty(len(gene_name)).astype(object)
		for d in swapping_group:
			if(type(d) is int):
				idx = np.where(degree == d)
			else: 
				idx = np.array([]).astype(int)
				for d_ in d:
					idx = np.concatenate((idx, np.where(degree == d_)[0]))
				
			genes = gene_name[idx]
			perm = np.random.permutation(genes)
			tot_perm[idx] = perm
		
		# save the permutations
		if(not os.path.exists(outdir)):
			os.makedirs(outdir)

		save_file(outdir + 'genes_' + str(j) + '.pkl', tot_perm)

	return 0
	

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
