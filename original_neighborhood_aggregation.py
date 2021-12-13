#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for constructing the list of SNPs
belonging to the 1-degree neighborhoods
on the PPI network. This is the step
for performing the 1-degree neighbourhood aggregation 
on the original network.

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

- 'data/mapping.pkl'				numpy vector with 2 columns, the SNPs and the
									genes they belong to

Command-line arguments:
--file_nw:    			      	          string; name of the file where the network is saved.
										  See above for the default.
--file_genes:    				          string; name of the file where the genes are saved.
										  See above for the default.
--file_mapping:							  string; name of the file where the gene-SNPs mapping
										  is saved. See above for the default.
--outdir:    				              string; name of the output folder.
'''
import pandas as pd
import numpy as np
from utils import *
import argparse
import os


def main(args):	
	# Setting parameters
	file_network, file_genes, file_mapping, outdir = args

	# LOADING INPUT FILES
	network   = load_file(file_network) # network
	gene_name = load_file(file_genes) # genes
	mapping   = load_file(file_mapping) # network

	neighbourhood_file(network, gene_name, mapping, outdir)
	
	return 0
	

def neighbourhood_file(A, gene_name, gene_snps, outdir):
	'''
	Function for performing the swapping of genes having the same 
	degree (or close)

	Input
	----------------
	A:          df containing the adjacency matrix of the network
	gene_name:  name of the genes
	gene_snps:  mapping of the genes and SNPs

	Output
	----------------
	'''
	if(not os.path.exists(outdir)):
		os.makedirs(outdir)

	snp_list = np.array(['snp', 'set']).astype(object)
	snp_list = snp_list.reshape(1, 2)
	for i, gene in enumerate(gene_name):
		connected_idx = (A[gene].values).astype(bool)
		snps = gene_snps[gene_snps[:, 1] == gene, 0]
		connected = gene_name[connected_idx]
		for conn in connected:
			snps = np.concatenate((snps, gene_snps[gene_snps[:, 1] == conn, 0]))

		snps = np.unique(snps)
		snp_list = np.concatenate((snp_list, np.c_[snps, np.full(len(snps), 'set_' + str(i))]))

	df = pd.DataFrame(snp_list)

	output = outdir + '/neighborhood_list.txt'
	df.to_csv(output, header = None, index = None, sep = ' ')


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
	parser.add_argument('--file_nw',      required = False, default = 'data/ppi.pkl')
	parser.add_argument('--file_genes',   required = False, default = 'data/gene_name.pkl')
	parser.add_argument('--outdir',       required = False, default = 'output/')
	parser.add_argument('--file_mapping', required = False, default = 'data/mapping.pkl')
	args = parser.parse_args()

	file_nw = args.file_nw
	file_genes = args.file_genes
	mapping = args.file_mapping
	outdir = args.outdir
	return file_nw, file_genes, mapping, outdir


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
