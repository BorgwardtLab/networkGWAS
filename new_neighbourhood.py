'''
Script for constructing the list of SNPs
belonging to the permuted 1-degree neighborhoods
given the list of permuted genes.
'''

import pandas as pd
import numpy as np
from IPython import embed
from utils import *
import argparse

def main(args):	
	# Setting parameters
	nperm = args

	# LOADING INPUT FILES
	network = load_file('data/ppi.pkl') # network
	gene_name = load_file('data/gene_name.pkl') # genes
	gene_snps = load_file('data/mapping.pkl') # gene-snp mapping
	
	neighbourhood_file(network, gene_name, gene_snps, nperm)
	
	return 0
	

def neighbourhood_file(A, gene_name, gene_snps, nperm):
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
	for j in range(nperm):
		perm = load_file('output/permutations/permuted_genes/genes_' + str(j) + '.pkl')
		snp_list = np.array(['snp', 'set']).astype(object)
		snp_list = snp_list.reshape(1, 2)
		for i, gene in enumerate(perm):
			# - i represents the position on the network
			connected_idx = (A[gene_name[i]].values).astype(bool)
			# - gene represents instead the snps to be assigned 
			# to the i-th gene. So, the position on the network 
			# and the respective neighbours remains. What changes
			# are the SNPs mapped on the gene
			snps = gene_snps[gene_snps[:, 1] == gene, 0]
			# For the connected genes, we don't need to obtain the
			# actual position on the network, but we need to obtain
			# the names of the new SNPs
			connected = perm[connected_idx]
			for conn in connected:
				snps = np.concatenate((snps, gene_snps[gene_snps[:, 1] == conn, 0]))


			snps = np.unique(snps)
			snp_list = np.concatenate((snp_list, np.c_[snps, np.full(len(snps), 'set_' + str(i))]))

		df = pd.DataFrame(snp_list)
		df.to_csv('output/permutations/nb_files/list_nb1_' + str(j) + '.txt', header = None, index = None, sep = ' ')


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
