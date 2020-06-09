import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm

from random import randint
import pickle

K = 10
RTBF = 0.10
MTBF = 0.10

# safe pickle loader
def pickleLoader(file, log):
	try:	
		with open(file, 'rb') as handle:
			res = pickle.load(handle)		
	except:
		print(("File %s not found." % file))
		return [], False
	return res, True

# safe pickle writer
def pickleWriter(obj, file, log):
	try:
		with open(file, 'wb') as handle:
			pickle.dump(obj, handle)		
	except:
		print("Saving %s failed." % str(obj)[:15])
		return False
	return True

# either logs into file or prints string
def logPrintIt(log, string):
	if log:
		logging.warning(string)
	else:
		print(string)


# creates k partitions of data
def kfolds(data):
	# shuffles data
	data = data.sample(frac=1)

	# initialise
	folds = []
	length = len(data)
	addition = 0
	size = 0

	# get sizes of partitions
	if length % K == 0:
		size = int(length / K)
	else:
		addition = length % K
		size = int((length - addition) / K)

	# get partitions
	for i in range(K):
		if i == K-1:
			folds.append(data[(i*size):])
		else:
			folds.append(data[(i*size):((i+1)*size)])

	return folds

# get train and test data
def trainTestData(data):
	# get folds
	folds = kfolds(data)
	# randomly chose a fold to be the test set
	i = randint(0, len(folds)-1)
	# initialise train
	train_data = pd.DataFrame()
	# append all folds except test fold to train data
	for x in range(len(folds)):
		if x != i:
			dat = pd.DataFrame(folds[x])
			train_data = pd.concat([train_data, dat])
	# return train data and test fold
	return train_data, folds[i]

# get target from model string
def getTarget(model):
	return model.split(" ~ ")[0]

# flattens a list
def flatten(lst):
	return [item for sublist in lst for item in sublist]

# get variables from model string
def getVars(model):
	var = []
	var_str = model.split(" ~ ")[1]
	if "+" not in var_str:
		return [var_str]
	else:
		return var_str.split(" + ")
	return var