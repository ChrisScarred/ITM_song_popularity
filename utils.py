import logging
import os.path

import pandas as pd
import numpy as np

from os import path
from random import randint

import pickle

import statsmodels.api as sm


K = 10
RTBF = 0.10
MTBF = 0.10

def pickleLoader(file, log):
	try:	
		with open(file, 'rb') as handle:
			res = pickle.load(handle)		
	except:
		print(("File %s not found." % file))
		return [], False
	return res, True

def pickleWriter(obj, file, log):
	try:
		with open(file, 'wb') as handle:
			pickle.dump(obj, handle)		
	except:
		print("Saving %s failed." % str(obj)[:15])
		return False
	return True

def logPrintIt(log, string):
	if log:
		logging.warning(string)
	else:
		print(string)


# creates k partitions of data, if shuffle, shuffles the order of data
def kfolds(data):
	data = data.sample(frac=1)

	folds = []
	length = len(data)
	addition = 0
	size = 0

	if length % K == 0:
		size = int(length / K)
	else:
		addition = length % K
		size = int((length - addition) / K)

	for i in range(K):
		if i == K-1:
			folds.append(data[(i*size):])
		else:
			folds.append(data[(i*size):((i+1)*size)])

	return folds


def trainTestData(data):
	folds = kfolds(data)
	i = randint(0, len(folds)-1)
	train_data = pd.DataFrame()
	for x in range(len(folds)):
		if x != i:
			dat = pd.DataFrame(folds[x])
			train_data = pd.concat([train_data, dat])
	return train_data, folds[i]

def getTarget(model):
	return model.split(" ~ ")[0]

def flatten(lst):
	return [item for sublist in lst for item in sublist]