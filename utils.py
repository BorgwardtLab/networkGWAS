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
