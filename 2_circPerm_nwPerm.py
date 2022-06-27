#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for performing networkGWAS's permutation
procedure, which comprises permutations on two 
levels, namely on both SNPs and network level.

1) permutation of the SNPs is performed through
a circular permutation, e.g., the SNPs are shifted 
in a circular manner.

2) degree-preserving permutation of the networks.

'''
import os
import csv
import argparse

import numpy as np
import pandas as pd

from utils import *


def main(args):
	
	# Setting the random seed for reproducibility of the results
	np.random.seed(args.seed)

	# Loading files
	gene2snps = load_file('{}/{}'.format(args.i, args.g2s))
	bim       = pd.read_csv('{}/{}'.format(args.i, args.bim), sep = '\t', 
						names = ['chrom', 'snp', '-', 'pos', 'a1', 'a2'])
	network   = load_file('{}/{}'.format(args.i, args.nw))

	# output folders creation
	if(not os.path.exists(args.onwdir)):
		os.makedirs(args.onwdir)

	if(not os.path.exists(args.onbdir)):
		os.makedirs(args.onbdir)

	# extracting SNPs names and genes names
	SNPs  = bim['snp'].values
	genes = np.array(network.index)
	
	# circular permutation & degree-preserving network permutation
	circular_permutations(SNPs, gene2snps, network, genes, args)


def neighbourhood_file(A, gene_name,  shifted_SNPs, gene_snps, j, args):
	'''
	Function for writing the neighborhood file where the 
	first column represents the neighborhoods, and the 
	second the SNPs belonging to the neighborhoods, as 
	in this example:

	snp           		set
	Chrom_1_23235 		AT1G01040
	Chrom_1_24466 		AT1G01040
	Chrom_2_11575494	AT1G01040
	Chrom_2_11575607 	AT1G01040
	Chrom_1_33481 		AT1G01060
	Chrom_3_22272106 	AT1G01060
	Chrom_5_26882325 	AT1G01060

	which means that the 1-hop neighborhood of AT1G01040
	comprises the SNPs named Chrom_1_23235, Chrom_1_24466,
	Chrom_2_11575494, and Chrom_2_11575607. To give another
	example, the SNPs Chrom_1_33481, Chrom_3_22272106, and
	Chrom_5_26882325 can be mapped onto AT1G01060 and its 
	1-hop neighbors.
	This function writes the files on the permuted settings.

	Input
	-----------
	A:             Ng x Ng pandas dataframe that represents the permuted
			       adjacency matrix of the PPI network. Index and columns'
			       names are the genes. 
	gene_name:	   Ng ordered genes, in the same way as the index/column of
				   the adjacency matrix.
	shifted_SNPs:  SNPs permuted according to circular permutation.
	gene_snps:     dictionary where the genes (Ng) are the keys and the
		 	       values are boolean vectors of size #SNPs: True 
		           if the SNP is mapped onto the gene that is the key.
	j:             permutation of index j.
	args:          arguments.
	
	Output
	-----------
	
	'''
	with open('{}/{}{}.txt'.format(args.onbdir, args.onb, str(j)), "w") as f:
		f.write('snp set\n')
		for i, gene in enumerate(gene_name):
			connected_idx = (A[gene].values).astype(bool)
			snps_idx = gene_snps[gene]
			# Finding the neighbors on the network
			connected = gene_name[connected_idx]
			for conn in connected:
				snps_idx = np.logical_or(snps_idx, gene_snps[conn])
			
			snps = shifted_SNPs[snps_idx]
			csv.writer(f, delimiter = ' ').writerows(np.c_[snps, np.full(len(snps), gene)])


def create_nw_df(connected_pairs, genes):
	'''
	Function for transforming the network in the form of connected 
	pairs to the adjacency matrix.
	
	Input
	------------
	connected_pairs: pairs of connected genes in
					 the permuted setting
	genes:           Ng ordered genes, in the 
					 same way as the index/column 
					 of the adjacency matrix.

	Output
	------------
	df:			     dataframe form of the permuted
					 network
	'''
	n = len(genes) # number of genes
	A = np.zeros((n, n)).astype(int)
	index1_ = []
	index2_ = []
	for k, pair in enumerate(connected_pairs):
		index1 = np.where(genes == pair[0])[0][0]
		index2 = np.where(genes == pair[1])[0][0]
		A[index1, index2] = 1

	A_symm = np.logical_or(A, A.T)
	df = pd.DataFrame(data = A_symm, columns = genes, index = genes)
	return df.copy()


def making_nw_triang_sup(nw):
	'''
	Transforming the adjacency matrix to a triangular
	superior adjacency matrix.

	Input
	---------
	nw:	     Ng x Ng pandas dataframe that represents the NON permuted
			 adjacency matrix of the PPI network. Index and columns'
			 names are the genes. 

	nw_up:   Ng x Ng pandas dataframe that represents the NON permuted
			 adjacency matrix of the PPI network. Index and columns'
			 names are the genes. In this case, the network is in the
			 triangular superior format.
	'''
	index_low = np.tril_indices(nw.shape[0])
	nw_vals = nw.values
	nw_vals[index_low] = 0
	nw_up = pd.DataFrame(data = nw_vals, columns = nw.columns, index = nw.columns)
	return nw_up


def find_connected_pairs(nw, genes):
	'''
	Function for rewriting the network in form of
	connected pairs.

	Input
	---------
	nw:		     Ng x Ng pandas dataframe that represents the NON permuted
			     adjacency matrix of the PPI network. Index and columns'
			     names are the genes. 
	genes:	     Ng ordered genes, in the same way as the index/column of
				 the adjacency matrix.

	Output
	---------
	pairs:       pairs of connected genes

	'''
	genes_nw = np.array(nw.columns)
	pairs = []
	for g in genes:
		connected = genes_nw[nw[g].values.astype(bool)]
		connected = np.setdiff1d(connected, g)
		for c in connected:
			pairs.append([g, c])
	
	return np.array(pairs)


def permuting_network(nw, genes, percentage, i):
	'''
	Degree-preserving permutation network.
	Pick randomly:
	1) 2 genes that are connected (A, B)
	2) 2 other genes that are connected (C, D)
	3) make sure A is not connected to C nor D
	4) make sure B is not connected to C nor D 
	5) remove connections between A-B and C-D
	6) connect A-C and B-D
	7) do this enough times, e.g. 50%

	Input
	----------
	nw:		     Ng x Ng pandas dataframe that represents the NON permuted
			     adjacency matrix of the PPI network. Index and columns'
			     names are the genes. 
	genes:	     Ng ordered genes, in the same way as the index/column of
				 the adjacency matrix.
	percentage:  percentage of edges to permute
	i:           permutation of index i.

	Output
	----------
	perm:		 Ng x Ng pandas dataframe that represents the permuted
			     adjacency matrix of the PPI network. Index and columns'
			     names are the genes. 

	'''
	nw_up = making_nw_triang_sup(nw.copy())
	connected_pairs = find_connected_pairs(nw_up, genes)
	n_edges = int(len(connected_pairs)*percentage)
	c = 0 # actual swaps counter
	while(c < n_edges):
		pair1_ix = np.random.choice(len(connected_pairs))
		pair2_ix = np.random.choice(len(connected_pairs))
		while(pair1_ix == pair2_ix): pair2_ix = np.random.choice(len(connected_pairs))
		pair1 = connected_pairs[pair1_ix].copy()
		pair2 = connected_pairs[pair2_ix].copy()
		genes = list(pair1) + list(pair2) # all the 4 gene analysed
		if(len(set(genes)) == 4):
			sub_nw = nw_up[genes].loc[genes]
			if(((sub_nw.sum().sum()) == 2)): # we can swap the edges!
				# updating the pairs of edges:
				connected_pairs[pair1_ix] = np.array([pair1[0], pair2[1]])
				connected_pairs[pair2_ix] = np.array([pair1[1], pair2[0]])
				c += 1
	
	perm = create_nw_df(connected_pairs, nw.columns)
	return perm


def circular_permutations(SNPs, gene_snps, nw, genes, args):
	'''
	The SNPs in the vector named "SNPs" are ordered according to their position on the genome.
	Thefore, we can just shift the SNPs of a random number, and then reassigning them to a gene
	and to a neighborhood, thus performing a circular permutation of the SNPs.

	Input
	---------
	SNPs:		SNPs ordered according to their position on the genome
	gene_snps:  dictionary where the genes (Ng) are the keys and the
		 	    values are boolean vectors of size #SNPs: True 
		        if the SNP is mapped onto the gene that is the key.
	nw:			Ng x Ng pandas dataframe that represents the adjacency
			    matrix of the PPI network. Index and columns' names
			    are the genes. 
	genes:		Ng ordered genes, in the same way as the index/column of
				the adjacency matrix.
	args:       arguments.

	Output
	---------
	'''
	N = SNPs.shape[0]
	SHIFTS = np.arange(1, N)
	MASK = np.array([True]*(N - 1))
	shift_vect = []
	for i in range(args.perm):
		shift = np.random.choice(SHIFTS[MASK])
		MASK[SHIFTS == shift] = False # not to repeat the same shift
		shift_vect.append(shift)
		shifted_SNPs = np.concatenate((SNPs[-shift:], SNPs[:(N - shift)]))
		# constructing gene-sets and neighborhood sets 
		# degree-preserving permutation scheme
		perm = permuting_network(nw, genes, args.alpha, i)
		save_file('{}/{}{}.pkl'.format(args.onwdir, args.onw, str(i)), perm)
		neighbourhood_file(perm, genes,  shifted_SNPs, gene_snps, i, args)
		if(i%10 == 0): print('{} permutations performed...'.format(i))


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------

	Output
	---------
	args.i:      input folder
	args.g2s:    name of the gene2snps mapping file (pickle); it is a  
		         dictionary where the genes (Ng) are the keys and the
		         values are boolean vectors of size #SNPs: True 
		 	     if the SNP is mapped onto the gene that is the key
	args.bim:    name of the Plink bim file; Plink bim file, with
		    	 #SNPs rows. 
	args.nw:     name of the adjacency matrix file (pickle); Ng x Ng 
		    	 pandas dataframe that represents the adjacency
		   		 matrix of the PPI network. Index and columns' names 
		    	 are the genes. 
	args.perm:   number of permutations
	args.alpha:  percentage of edges to permute
	args.onwdir: output folder for the permuted
				 networks
	args.onbdir: output folder for the permuted
				 neighborhoods
	args.onw:    base filename for the permuted
				 networks
	args.onb:    base filename for the permuted
				 neighborhoods
	
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--i',     required = False, default = 'data',    
							help = 'input folder')
	parser.add_argument('--g2s',   required = False, default = 'gene_snps_index.pkl', 
							help = 'mapping between the genes and the snps')
	parser.add_argument('--bim',   required = False, default = 'genotype.bim', 
							help = 'bim file, where to get the snp names')
	parser.add_argument('--nw',    required = False, default = 'PPI_adj.pkl', 
							help = 'adjacency matrix of the PPI network')
	parser.add_argument('--perm',  required = False, default = 300, type = int, 
							help = 'number of permutations')
	parser.add_argument('--alpha', required = False, default = 0.5, type = float, 
							help = 'percentage of edges to permute')
	parser.add_argument('--seed',  required = False, default = 42, type = int, 
							help = 'random seed for reproducibility')
	parser.add_argument('--onwdir',   required = False, 
							default = 'results/settings/permutations/networks/', 
							help = 'output folder the permuted networks')
	parser.add_argument('--onbdir',   required = False, 
							default = 'results/settings/permutations/neighborhoods/', 
							help = 'output folder for the permuted neighborhoods')
	parser.add_argument('--onw',   required = False, default = 'nw_', 
							help = 'base name for the permuted networks')
	parser.add_argument('--onb',   required = False, default = 'nbs_', 
							help = 'base name for the permuted neighborhoods')
	args = parser.parse_args()
	return args



if __name__ == '__main__':
	args = parse_arguments()
	main(args)


