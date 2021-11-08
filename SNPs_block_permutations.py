#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for saving the permuted genes;
the permutation is done by following
the SNPs-block-permutation strategy

File in input:
- 'data/snps_list.pkl':   				  numpy vector containing the names of the SNPs 
										  included in the network

- 'data/mapping.pkl':					  numpy vector with 2 columns, the SNPs and the
										  genes they belong to

- 'data/gene_name.pkl':   				  numpy vector containing the names of the genes 
										  included in the network

- 'output/permutations/':  				  string where to save the permuted mapping
										 

Command-line arguments:
--nperm:    				              integer; it's the number of permutation to perform
--blocksize:    				          integer; it's the size of the blocks.
--file_snps:    				          string; name of the file where the SNPs are saved.
										  See above for the default.
--file_mapping:    				          string; name of the file where the gene-SNPs mapping
										  is saved. See above for the default.
--file_genes:    				          string; name of the file where the genes are saved.
										  See above for the default.
--outdir:    				              string; name of the output folder.
'''

import numpy as np
import pandas as pd
import argparse
from utils import *
import re
import os

def main(args):	
	nperm, blocksize, file_snps, file_mapping, file_genes, outdir = args

	# LOADING INPUT FILES
	mapping   = load_file(file_mapping) # network
	gene_name = load_file(file_genes) # genes
	snps      = load_file(file_snps)


	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	# Calling the function for performing the 
	# SNPs-block permutation technique
	lengths = obtain_gene_length(mapping, gene_name)
	permutations(gene_name, nperm, lengths, snps, blocksize, outdir)
	
	return 0
	

def obtain_gene_length(gene_snps, gene_name):
	'''
	function for obtaining the number of SNPs 
	per each gene
	'''	
	lengths = {}
	for gene in gene_name:
		l = (gene_snps[:, 1] == gene).sum()
		lengths[gene] = l
	
	return lengths


def blocks_permutations(blocks):
	'''
	function for permuting the blocks
	'''
	block_id = np.array(list(blocks.keys()))
	permuted = np.random.permutation(block_id)
	snps = np.array([])
	for block in permuted:
		snps = np.concatenate((snps, blocks[block]))

	return snps


def blocks_definition(snps, block_size):
	'''
	function for permuting the SNPs in blocks;
	the size is defined by the variable "block_size"
	'''
	n = len(snps)
	blocks_number = int(n/block_size)
	i1, blocks = 0, {}
	for b in range(blocks_number):
		i2 = i1 + block_size
		blocks[b] = snps[i1:i2]
		i1 = i2
	
	left = n - (blocks_number*block_size)
	if(left > 0):
		blocks[b + 1] = snps[-left:]
	
	return blocks


def permutations(gene_name, nperm, lengths, snps, block_size, outdir):
	'''
	function for permuting the SNPs
	'''
	blocks = blocks_definition(snps, block_size)
	for j in range(nperm): # per each permutation
		permuted = blocks_permutations(blocks)
		gene_snps = {}
		n1 = 0
		check = np.array([])
		for gene in gene_name:
			n2 = n1 + lengths[gene]
			gene_snps[gene] = permuted[n1:n2]
			
			# adjusting n1 for the next iteration
			n1 = n2
		

		outdir_tot = outdir + '/permuted_mapping_' + str(block_size) + '/'
		if(not os.path.exists(outdir_tot)):
			os.mkdir(outdir_tot)

		save_file(outdir_tot + '/mapping_' + str(j) + '.pkl', gene_snps)

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
	parser.add_argument('--file_snps',    required = False, default = 'data/snps_list.pkl')
	parser.add_argument('--file_mapping', required = False, default = 'data/mapping.pkl')
	parser.add_argument('--file_genes',   required = False, default = 'data/gene_name.pkl')
	parser.add_argument('--outdir',       required = False, default = 'data/output/permutations/')
	args = parser.parse_args()

	nperm = args.nperm
	blocksize = args.blocksize
	file_snps = args.file_snps
	file_mapping = args.file_mapping
	file_genes = args.file_genes
	outdir = args.outdir
	return nperm,  blocksize, file_snps, file_mapping, file_genes, outdir


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
