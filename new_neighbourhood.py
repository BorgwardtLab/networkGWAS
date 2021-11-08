#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for constructing the list of SNPs
belonging to the permuted 1-degree neighborhoods
given the list of permuted genes. This is the step
for performing the 1-degree neighbourhood aggregation 
on the permuted network.

Inputs:
- 'data/ppi.pkl': 		  		  	PPI network in form of pandas DataFrame. The
								  	keys of the data frame are the names of the 
									genes included in the network. The values of
									the dataframe is the non-weighted adjacency 
									matrix of the PPI network, e.g. 1 when there is
								 	an edge between the 2 genes, and 0 when there is
								  	no edge, e.g. no interaction

- 'data/gene_name.pkl':   		  	numpy vector containing the names of the genes 
						  		  	included in the network

- 'output/permutations':            string where to save the 1-degree aggregations on
									the permuted networks.

Command-line arguments:
--nperm:    				              integer; it's the number of permutation to perform
--blocksize:    				          integer; it's the size of the blocks.
--file_nw:    			      	          string; name of the file where the network is saved.
										  See above for the default.
--file_genes:    				          string; name of the file where the genes are saved.
										  See above for the default.
--outdir:    				              string; name of the output folder.
'''
import pandas as pd
import numpy as np
from utils import *
import argparse
import os


def main(args):	
	# Setting parameters
	nperm, blocksize, file_network, file_genes, outdir = args

	# LOADING INPUT FILES
	network   = load_file(file_network) # network
	gene_name = load_file(file_genes) # genes

	neighbourhood_file(network, gene_name, nperm, outdir, blocksize)
	
	return 0
	

def neighbourhood_file(A, gene_name, nperm, outdir, block_size):
	'''
	Function for performing the swapping of genes having the same 
	degree (or close)

	Input
	----------------
	A:          df containing the adjacency matrix of the network
	gene_name:  name of the genes
	gene_snps:  mapping of the genes and SNPs
	nperm:      number of permutations

	Output
	----------------
	'''
	# save the permutations
	outdir_tot = outdir + '/neighborhoods_' + str(block_size) 
	if(not os.path.exists(outdir_tot)):
		os.makedirs(outdir_tot)



	for j in range(nperm):
		snp_list = np.array(['snp', 'set']).astype(object)
		snp_list = snp_list.reshape(1, 2)
		gene_snps = load_file(outdir + '/permuted_mapping_' + str(block_size) + '/mapping_' + str(j) + '.pkl')
		for i, gene in enumerate(gene_name):
			connected_idx = (A[gene].values).astype(bool)
			snps = gene_snps[gene]
			connected = gene_name[connected_idx]
			for conn in connected:
				snps = np.concatenate((snps, gene_snps[conn]))

			snps = np.unique(snps)
			snp_list = np.concatenate((snp_list, np.c_[snps, np.full(len(snps), 'set_' + str(i))]))

		df = pd.DataFrame(snp_list)

		df.to_csv(outdir + '/neighborhoods_' + str(block_size) +  '/list_nb1_' + str(j) + '.txt', header = None, index = None, sep = ' ')


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
	parser.add_argument('--nperm',        required = False, default = 1000, type = int)
	parser.add_argument('--blocksize',    required = False, default = 50,   type = int)
	parser.add_argument('--file_nw',   required = False, default = 'data/ppi.pkl')
	parser.add_argument('--file_genes',   required = False, default = 'data/gene_name.pkl')
	parser.add_argument('--outdir',       required = False, default = 'output/permutations')
	args = parser.parse_args()

	nperm = args.nperm
	blocksize = args.blocksize
	file_nw = args.file_nw
	file_genes = args.file_genes
	outdir = args.outdir
	return nperm,  blocksize, file_nw, file_genes, outdir


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
