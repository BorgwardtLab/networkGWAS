#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for generating the neighborhood file on
the non-permuted setting. The neighborhood file
consists in two columns, for example:

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
'''
import os
import csv
import argparse

import numpy as np
import pandas as pd

from utils import *


def main(args):

	# Loading files
	gene2snps = load_file('{}/{}'.format(args.i, args.g2s))
	bim       = pd.read_csv('{}/{}'.format(args.i, args.bim), sep = '\t', 
						names = ['chrom', 'snp', '-', 'pos', 'a1', 'a2'])
	network   = load_file('{}/{}'.format(args.i, args.nw))

	# output folders creation
	if(not os.path.exists(args.o)):
		os.makedirs(args.o)

	# 1-hop neighborhoods construction
	neighbourhood_file(gene2snps, bim, network, args)

		
def neighbourhood_file(gene2snps, bim, A, args):
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

	Input
	-----------
	gene2snps: dictionary where the genes (Ng) are the keys and the
		 	   values are boolean vectors of size #SNPs: True 
		       if the SNP is mapped onto the gene that is the key
	bim: 	   Plink bim file, with #SNPs rows
	A:         Ng x Ng pandas dataframe that represents the adjacency
		  	   matrix of the PPI network. Index and columns' names 
		 	   are the genes. 
	args:      arguments
	
	Output
	-----------
	'''
	SNPs  = bim['snp'].values
	genes = np.array(A.index)
	# open the output neighborhood file
	with open('{}/{}'.format(args.o, args.nbs), "w") as f:
		f.write('snp set\n')
		for gene in genes:
			conn_idx = (A[gene].values).astype(bool)
			snps_idx = gene2snps[gene]
			# Finding the neighbors on the network
			connected = genes[conn_idx]
			for conn in connected:
				snps_idx = np.logical_or(snps_idx, gene2snps[conn])
			
			snps = SNPs[snps_idx]
			# writing the neighborhood
			csv.writer(f, delimiter = ' ').writerows(np.c_[snps, np.full(len(snps), gene)])


def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------

	Output
	---------
	args.i:    input folder
	args.o:	   output folder
	args.g2s:  name of the gene2snps mapping file (pickle); it is a  
		  	   dictionary where the genes (Ng) are the keys and the
		       values are boolean vectors of size #SNPs: True 
		       if the SNP is mapped onto the gene that is the key
	args.bim:  name of the Plink bim file; Plink bim file, with
		       #SNPs rows. 
	args.nw:   name of the adjacency matrix file (pickle); Ng x Ng 
		       pandas dataframe that represents the adjacency
		       matrix of the PPI network. Index and columns' names 
		       are the genes. 
	args.nbs:  name of the output text file where to save the 1-hop
		       neighborhoods in the form of name of SNPs 
	
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--i',   required = False, default = 'data',    
							help = 'input folder')
	parser.add_argument('--o',   required = False, default = 'results/settings', 
							help = 'output folder')
	parser.add_argument('--g2s', required = False, default = 'gene_snps_index.pkl', 
							help = 'mapping between the genes and the snps')
	parser.add_argument('--bim', required = False, default = 'genotype.bim', 
							help = 'bim file, where to get the snp names')
	parser.add_argument('--nw',  required = False, default = 'PPI_adj.pkl', 
							help = 'adjacency matrix of the PPI network')
	parser.add_argument('--nbs', required = False, default = 'neighborhoods.txt', 
							help = 'name of the file containing the 1-hop\
							neighborhoods on the non-permuted setting\
							to save in the output folder')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)



