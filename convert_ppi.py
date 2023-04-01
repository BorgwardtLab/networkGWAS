#!/usr/bin/env python
# Author: Giulia Muzio

'''
Script for converting the network from
edge format to adjacency matrix format:

Edge format:

gene1 gene2
gene1 gene4
gene2 gene3


Adjacency matric format:

		gene1 gene2 gene3 gene4
gene1	0		1 	  0		1
gene2	1 		0	  1 	0
gene3	0		1 	  0     0
gene4	1 		0     0     0

'''
import argparse

import numpy as np
import pandas as pd

from utils import *


def main(args):
	network_preprocessing(args.i, args.o)


def network_preprocessing(input, output):
	'''
	Function for preprocessing the network database

	Input
	--------
	input:    filename of the network in edge format
	ouput:	  filename of the network in adjacency matrix format

	Output
	--------
	'''
	
	# save the two columns
	protein1 = []
	protein2 = []
	combined_score = []
	with open(input) as f:
		for line in f:
			parts = line.rstrip().split()
		
			p1 = parts[0]
			p2 = parts[1]
			if (p1 != p2): # not considering self-interactions
				protein1.append(p1)
				protein2.append(p2)

	f.close()
	
	# a dataframe with the adjacency matrix of the PPI network 
	proteins = np.unique(np.concatenate((protein1, protein2)))
	n = len(proteins)
	df_ = pd.DataFrame(data = np.zeros((n, n)).astype(int), columns = proteins, index = proteins)
	for p1, p2 in zip(protein1, protein2):
		df_[p1][p2] = 1
		df_[p2][p1] = 1 # make sure it's symmetric: we do not consider directionality of the interactions
	
	save_file(output, df_)
	return 0




def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------

	Output
	---------
	args.i:    input file: ppi network in edges format
	args.o:	   output file: ppi network in adjacency matrix format
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--i',   required = False, default = 'data/ppi_edges.txt',    
							help = 'input file')
	parser.add_argument('--o',   required = False, default = 'data/PPI_adj.pkl', 
							help = 'output file')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)



