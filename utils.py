#!/usr/bin/env python
# Author: Giulia Muzio

import pickle
import numpy as np

def save_file(filename, data):
	'''
	Function for saving file in the pickle format

	Input
	-----------
	filename:  name of the output file
	data:      data to save

	Output
	----------
	'''
	with open(filename, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_file(filename):
	'''
	Function for loading pickle format files

	Input
	-----------
	filename:  name of the output file
	
	Output
	----------
	pickle.load(f):    the data
	'''
	with open(filename, 'rb') as f:
		return pickle.load(f)


def performance(pos_snps, causal, p):
	'''
	Function for evaluating the results of the methods
	True and false positives, true and false negatives,
	precision and recall.

	Input
	-----------
	pos_snps:    prediction
	causal:      ground truth
	p:           number of SNPs, features

	Output
	-----------
	'''
	tp = len(np.intersect1d(pos_snps, causal))
	fp = len(np.setdiff1d(pos_snps, causal))
	fn = len(causal) - tp
	tn = p - (tp + fp + fn)

	print('\nPERFORMANCE:')
	print('TP: ' + str(tp) )
	print('FP: ' + str(fp) )
	print('TN: ' + str(tn) )
	print('FN: ' + str(fn) )
	print('-----------------')
	print('Precision: ' + str(np.round(tp/(tp + fp), 3)))
	print('Recall: ' + str(np.round(tp/(tp + fn), 3)))
	return 0