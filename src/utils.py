import logging
import pickle
import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from textwrap import wrap

K = 10
RTBF = 0.10
MTBF = 0.10


def print_scpplot(x, y, data, prediction, axes=[], mdl=""):
	if axes == []:
		sns.scatterplot(x = data[x], y = data[y]).set(title = "\n".join(wrap('Relationship between %s and %s including the linear model %s' % (x, y, mdl), 60)))
		sns.lineplot(x = data[x], y = prediction, color = '#bd1d00', estimator=None)
	else:
		sns.scatterplot(x = data[x], y = data[y], ax = axes).set(title = "\n".join(wrap('Relationship between %s and %s including linear model %s' % (x, y, mdl), 60)))
		sns.lineplot(x = data[x], y = prediction, color = '#bd1d00', ax = axes, estimator=None)

def print_scplot(x, y, data, axes=[]):
	if axes == []:
		sns.scatterplot(x = data[x], y = data[y]).set(
			title = "\n".join(wrap(('Relationship between %s and %s' % (x, y)), 60)))
	else:
		sns.scatterplot(x = data[x], y = data[y], ax = axes).set(
			title = "\n".join(wrap(('Relationship between %s and %s' % (x, y)), 60)))
		
		
def make_func_plots(x_vars, y_vars, col, func, data, prediction=[], mdl=""):
	if len(x_vars) == 1 and len(y_vars)==1:
		plt.figure()
		if prediction==[]:
			func(x_vars[0], y_vars[0], data)
		else:
			func(x_vars[0], y_vars[0], data, prediction[0], mdl)
		plt.show()
	number = len(x_vars)*len(y_vars)
	
	if not number%col==0:
		number += (col - (number%col))
	
	sub = int(number/col)

	fig, axs = plt.subplots(sub, col, figsize=((7.5*col),(7.5*sub)))

	if col == 1 or sub == 1:
		coo = 0
		for x in range(len(x_vars)):
			for y in range(len(y_vars)):
				if prediction==[]:
					func(x_vars[x], y_vars[y], data, axs[coo])
				else:
					func(x_vars[x], y_vars[y], data, prediction[coo], axs[coo], mdl)
				coo += 1

	else:
		i = 0
		for x in range(len(x_vars)):
			for y in range(len(y_vars)):
				no = (x+1)*(y+1)-1
				y_coo = int(no%col)
				x_coo = int((no - (no%col))/col)
				if prediction==[]:
					func(x_vars[x], y_vars[y], data, axs[x_coo, y_coo])
				else:
					func(x_vars[x], y_vars[y], data, prediction[i], axs[x_coo, y_coo], mdl)
				i += 1
	plt.tight_layout()
	plt.show()

def print_all_model_stats(model_str, data):
	model = sm.formula.ols(formula=model_str, data=data)
	model_fitted = model.fit()
	print(model_fitted.summary())

	xs = getVars(model_str)
	y = [getTarget(model_str)]
	p = model_fitted.predict(data)
	preds = [p for i in range(len(xs))]
	make_func_plots(xs, y, 2, print_scpplot, data, prediction=preds, mdl=model_str)

	print_stat_plots(model_str, data)

def print_stat_plots(model_str, data):
	# gets and fit the model
	model = sm.formula.ols(formula=model_str, data=data)
	model_fitted = model.fit()

	# gets intercept and variable info
	intercept = model_fitted.params[0]
	variables = getVars(model_str)

	# gets target
	target = getTarget(model_str)

	# gets scores
	predscores = model_fitted.predict(data)
	targets = data[target]

	resscores = targets - predscores

	# creates figures
	fig, axs = plt.subplots(2, 2, figsize=(15,15))

	# abs residuals
	sns.scatterplot(x = predscores, y = np.abs(resscores), ax = axs[0,0]).set(
		title = "\n".join(wrap(('Absolute residuals against predicted values for model %s' % model_str), 60)), 
		xlabel = 'Predicted scores', 
		ylabel = 'Residuals')

	# histogram
	sns.distplot(resscores, bins = 15, ax = axs[0, 1]).set(
		title = "\n".join(wrap(('Histogram of residual scores for model %s' % model_str), 60)),
		xlabel = 'Residual scores', 
		ylabel = 'Probability')


	scipy.stats.probplot(resscores, plot = axs[1, 0])
	axs[1, 0].get_lines()[0].set_markerfacecolor('#c5c5d6')
	axs[1, 0].get_lines()[0].set_markeredgecolor('#c5c5d6')

	# residuals
	sns.scatterplot(x = list(range(0, len(data[target]))), y = resscores, ax = axs[1,1]).set(
		title = "\n".join(wrap(('Residuals against order of collection for model %s' % model_str), 60)),
		xlabel = 'Order of collection', 
		ylabel = 'Residuals')

	plt.tight_layout()
	plt.show()

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

def print_p_vals(targets, variables, data):
	for target in targets:
		print("-"*10)
		for var in variables:
			formula_str = target + " ~ " + var
			model = sm.formula.ols(formula=formula_str, data=data)
			model_fitted = model.fit()
			print("p-value of %s: %.4f" % (var, model_fitted.pvalues[1]))