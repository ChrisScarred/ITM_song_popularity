import statsmodels.api as sm
import numpy as np
from itertools import combinations
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from textwrap import wrap


# files to save data in
root = Path(".")

BS_MODELFILE = root / "data" / 'bs_models.pickle'
BS_MAPEFILE = root / "data" / 'bs_mape.pickle'
BS_RFILE = root / "data" / 'bs_r.pickle'

FS_MODELFILE = root / "data" / 'fs_models.pickle'
FS_MAPEFILE = root / "data" / 'fs_mape.pickle'
FS_RFILE = root / "data" / 'fs_r.pickle'

SVM_MODELFILE = root / "data" / 'svm_models.pickle'
SVM_MAPEFILE = root / "data" / 'svm_mape.pickle'
SVM_RFILE = root / "data" / 'svm_r.pickle'

BSVM_MODELFILE = root / "data" / 'bsvm_models.pickle'
BSVM_MAPEFILE = root / "data" / 'bsvm_mape.pickle'
BSVM_RFILE = root / "data" / 'bsvm_r.pickle'

# parameters
EPOCHS = 10
R_THRESHOLD = 0.05
MAPE_THRESHOLD = 0.20

# folder for summary saving
SUM_FOLDER = 'summaries/'

'''
The critical part of the module that computes the best models with different strategies
Initialised with data and log indicator
'''
class Analyses:
	def __init__(self, data, log):
		self.data = data
		self.log = log

	# plots models and variables in to_plot
	def make_plots(self, to_plot):
		# gets vars
		x_vars = to_plot[0]
		y_vars = to_plot[1]
		colin_vars = to_plot[2]
		statplots = to_plot[3]
		# gets combinations for colinearity analysis
		colin_pairs = combinations(colin_vars, 2)

		# generates scatter plot of every x var wrt every y var
		for x in x_vars:
			for y in y_vars:
				self.scplot(x, y)

		# generates scatter plots for all combinations of colin vars
		for comb in colin_pairs:
			x = comb[0]
			y = comb[1]
			self.scplot(x, y)

		# generates stats plots for every model queried
		for model in statplots:
			self.statPlots(model)

		print("Done generating plots.")

	# generates scatter plot out of data at x and y
	def scplot(self, x, y):
		plt.figure()
		sns.scatterplot(x = self.data[x], y = self.data[y]).set(title = "\n".join(wrap(('Relationship between %s and %s' % (x, y)), 60)))
		name = 'plots/scatters/'+x+"_againts_"+y+".png"
		plt.savefig(name)
		plt.close()

	# generates scatter plot out of data at x and y and draws prediction line prediction over it
	def scPredPlot(self, x, y, prediction):
		sns.scatterplot(x = self.data[x], y = self.data[y]).set(title = "\n".join(wrap('Relationship between %s and %s including linear model' % (x, y), 60)))
		sns.lineplot(x = self.data[x], y = prediction, color = '#bd1d00')
		name = 'plots/with_prediction/'+x+"_againts_"+y+"_with_prediction.png"
		plt.savefig(name)
		plt.close()

	# generates stats plots from a given model string
	def statPlots(self, model_str):
		# gets and fit the model
		model = sm.formula.ols(formula=model_str, data=self.data)
		model_fitted = model.fit()

		# gets intercept and variable info
		intercept = model_fitted.params[0]
		variables = getVars(model_str)

		# gets target
		target = getTarget(model_str)

		# gets scores
		predscores = model_fitted.predict(self.data)
		targets = self.data[target]

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
		sns.scatterplot(x = list(range(0, len(self.data[target]))), y = resscores, ax = axs[1,1]).set(
			title = "\n".join(wrap(('Residuals against order of collection for model %s' % model_str), 60)),
			xlabel = 'Order of collection', 
			ylabel = 'Residuals')

		# saving
		namestr = model_str.replace(" ", "_")
		name = 'plots/statplots/statplot_'+namestr+".png"
		plt.tight_layout()
		plt.savefig(name)
		plt.close()

	# saves summaries as txt file for every model string in list models
	def printSummaries(self, models):
		for string in models:			
			mdl = sm.formula.ols(formula = string, data = self.data)
			model_fitted = mdl.fit()
			res = str(model_fitted.summary())

			fname = SUM_FOLDER + string.replace(" ", "_")+".txt"
			f = open(fname, "w")
			f.write(res)
			f.close()

		print("Done saving summaries.")

	# prints (or logs) custom queried models
	def printCustom(self, models):
		mapes, rs = self.getPerformance(models)
		self.printModels(models, mapes, rs, string="Stats for custom models:")
		return models

	# generates all single variable models
	def singleVarModels(self):	
		targets, variables = self.getTargVars()
		
		models = []

		# generates the model string for every one-variable and target combination
		for target in targets:			
			for var in variables:
				model = target + " ~ " + var
				models.append(model)
		
		# gets the performance of models
		mape, r = self.getPerformance(models)
		
		# returns models along with performance
		return models, mape, r

	# chooses sufficiently well-performing single var models from lists of models, mapes and rs
	def bestSingleVars(self, models, mapes, rs):
		suff_models, mape_new, r_new = self.getSufficient(models, mapes, rs)
		return suff_models, mape_new, r_new
	
	# chooses sufficiently well-performing single var models from lists of models, mapes and rs
	def getSufficient(self, i_models, i_mape, i_r):
		models = []
		mapes = []
		rs = []

		# zip for easier manipulation
		zipped = zip(i_models, i_mape, i_r)

		# append to result if mape or r is better than their threshold
		for (model, mape, r) in zipped:
			if (mape < MAPE_THRESHOLD) or (r > R_THRESHOLD):
				models.append(model)
				mapes.append(mape)
				rs.append(r)
		
		# prints (logs) info if no sufficient models found
		if len(models) == 0:
			logPrintIt(self.log, "No sufficiently well-performing model with single variable found.")

		# returns sufficient models with their performance
		return models, mapes, rs

	# prints or logs basic information about model with model string model and MAPE mape and r r
	def printModel(self, model, mape, r):
		logPrintIt(self.log, ("Model '%s' with MAPE of %.3f and r^2 of %.3f" % (model, mape, r)))

	# prints models with model strings in the list models, MAPEs in the list mape, rs in the list r along with informative string string
	def printModels(self, models, mape, r, string="Queried models:"):
		s = "----------------------------------"
		string = "\n" + s + "\n" + string + "\n" + s
		logPrintIt(self.log, string)

		for i in range(len(models)):
			self.printModel(models[i], mape[i], r[i])


	# obtains models by either reading them from saved data in files models_file, mapes_file and rs_file or generating them via function func
	def modelGetter(self, models_file, mapes_file, rs_file, func, string, fail_msg):
		# safe load
		models, r1 = pickleLoader(models_file, self.log)
		mapes, r2 = pickleLoader(mapes_file, self.log)
		rs, r3 = pickleLoader(rs_file, self.log)

		# if could not load, generate
		if not r1 or not r2 or not r3:
			models, mapes, rs = func

			# safe write
			if len(models) > 0:
				pickleWriter(models, models_file, self.log)
				pickleWriter(mapes, mapes_file, self.log)
				pickleWriter(rs, rs_file, self.log)

		# print (log) if any models generated
		if len(models) > 0:
			self.printModels(models, mapes, rs, string=string)
		# print (log) fail message otherwise
		else:
			logPrintIt(self.log, fail_msg)
		# returns models along with their performance
		return models, mapes, rs

	# calls modelGetter with different algorithms to get best performing models using multiple methods
	def bestModelSearch(self):
		# generates all single variable models
		models, mapes, rs = self.modelGetter(SVM_MODELFILE, SVM_MAPEFILE, SVM_RFILE, self.singleVarModels(), 
			"All single variable models:", "No single variable model found.")

		# gets the best performing single variable models from data above
		models2, mapes2, rs2 = self.modelGetter(BSVM_MODELFILE, BSVM_MAPEFILE, BSVM_RFILE, self.bestSingleVars(models, mapes, rs), 
			"Best single variable models:", "No well-performing single variable model found.")

		# gets the best model(s) obtained via forward selection using already generated single var models as a base
		models3, mapes3, rs3 = self.modelGetter(FS_MODELFILE, FS_MAPEFILE, FS_RFILE, self.forward_selection(models, mapes, rs), 
			"Best models obtained via forward selection:", "No well-performing model found via forward selection.")
		
		# gets the best model(s) via backward selection
		models4, mapes4, rs4 = self.modelGetter(BS_MODELFILE, BS_MAPEFILE, BS_RFILE, self.backward_selection(), 
			"Best models obtained via backward selection:", "No well-performing model found via backward selection.")
		
		# combine best model string into one list
		models.extend(models3)
		models.extend(models4)

		# return the final list
		return models 

	# gets the best performing model based on a criterion; lower True if criterion should be as small as possible
	def getBest(self, models, criterion, lower):			
		zipped = zip(models, criterion)		
		res = sorted(zipped, key=lambda x: x[1])

		if lower:
			return res[0][0], res[0][1]

		return res[-1][0], res[-1][1]

	# returns targets and variables from the data
	def getTargVars(self):
		no_includes = ["name", "id"]
		targets = ["popularity_rel", "popularity_abs"]
		variables = [c for c in self.data.columns if c not in targets and c not in no_includes]
		return targets, variables

	# performs forward selection based on MAPE and r^2
	def forward_selection(self, singleVarModels, mape, r):
		# gets best single var models		
		best_mape_model, best_mape = self.getBest(singleVarModels, mape, True)
		best_r_model, best_r = self.getBest(singleVarModels, r, False)
		# gets targets and vars
		targets, variables = self.getTargVars()

		# while there is an improvement, do
		gain_mape = True
		gain_r = True
		while gain_mape or gain_r:
			mapes = []
			rs = []
			mape_models = []
			r_models = []

			# for every variable
			for var in variables:
				# if it is not in the current best mape model, try whether adding it improves the model
				if var not in best_mape_model and gain_mape:
					model = best_mape_model + " + " + var

					# get performance, compare
					mape, r = self.getPerformance([model])
					mape = mape[0]
					# if new mape is better, add to the list of potential new best mape models
					if mape < best_mape:
						mape_models.append(model)
						mapes.append(mape)

				# if it is not in the current best r model, try whether adding it improves the model
				if var not in best_r_model and gain_r:
					model = best_r_model + " + " + var

					# get performance, compare
					mape, r = self.getPerformance([model])
					r = r[0]
					# if new r is better, add to the list of potential new best r models
					if r > best_r:
						r_models.append(model)
						rs.append(r)

			# if no better mape model than current found, mape improvement no longer possible
			if mapes == [] or mape_models == []:
				gain_mape = False
			# otherwise get the new best mape model
			else:
				best_mape_model, best_mape = self.getBest(mape_models, mapes, True)	

			# if no better r model than current found, r improvement no longer possible
			if rs == [] or r_models == []:
				gain_r = False
			# otherwise get the new best r model
			else:
				best_r_model, best_r = self.getBest(r_models, rs, False)				
				
		# obtain missing data for the two best models
		m, r_for_mape = self.getPerformance([best_mape_model])
		mape_for_r, n = self.getPerformance([best_r_model])

		r_for_mape = r_for_mape[0]
		mape_for_r = mape_for_r[0]

		# construct the list of final models
		final_models = [best_mape_model, best_r_model]		
		final_mape = [best_mape, mape_for_r]
		final_r = [r_for_mape, best_r]

		# return the final best model along with their performance
		return final_models, final_mape, final_r

	# removes variable var from model string model such that the string remains a valid model string
	def getBackString(self, model, var):
		# remove var
		str_new = model.replace(var, "")
		# if var was the first, then there is ~ + instead of ~
		str_new = str_new.replace("~  + ", "~ ")
		# if var was not the first one, there might be two pluses in a row, so replace by one
		str_new = str_new.replace("+  + ", "+ ")
		# get rid of unnecessary whitespaces
		str_new = str_new.strip()

		# if var was the last one, string ends with +, so remove that and get rid of whitespaces
		if str_new.endswith("+"):
			str_new = str_new[:-1].strip()

		# if the var was not the only one in str, get the new models performance and return
		if not(str_new.split(" ~ ")[-1] == ""):		
			model = str_new					
			mape, r = self.getPerformance([model])
			mape = mape[0]
			r = r[0]
			return model, mape, r 

		# if no valid model remains, return false
		else:
			return False, False, False

	# performs backward selection of the best model based on MAPE and r^2
	def backward_selection(self):
		# get targets and variables
		targets, variables = self.getTargVars()

		# create a model string containing every variable for every target
		models = []
		for target in targets:
			mdl_str = target
			mdl_str += " ~ "
			for i in range(len(variables)):
				if i == len(variables) - 1:
					mdl_str += variables[i]
				else:
					mdl_str += variables[i]
					mdl_str += " + "
			models.append(mdl_str)

		# get performance of the two models
		mape, r = self.getPerformance(models)

		# choose the best among them
		best_mape_model, best_mape = self.getBest(models, mape, True)
		best_r_model, best_r = self.getBest(models, r, False)

		# while improvement possible, do
		gain_mape = True
		gain_r = True
		while gain_mape or gain_r:
			mapes = []
			rs = []
			mape_models = []
			r_models = []

			# for every variable
			for var in variables:
				targ = getTarget(best_mape_model)
				# check if it is in the current best mape model
				if var in best_mape_model and gain_mape:
					# if so, get performance of the model without it
					new_model, new_mape, new_r = self.getBackString(best_mape_model, var)
					# if valid model cannot be constructed without it, pass
					if not new_model:
						pass
					# else find out if performance better, if so add to the list of potentional new best mape models
					elif new_mape < best_mape:
						mape_models.append(new_model)
						mapes.append(new_mape)

				targ = getTarget(best_r_model)
				# check if it is in the current best r model
				if var in best_r_model and gain_r:
					# if so, get performance of the model without it
					new_model, new_mape, new_r = self.getBackString(best_r_model, var)
					# if valid model cannot be constructed without it, pass
					if not new_model:
						pass
					# else find out if performance better, if so add to the list of potentional new best mape models
					elif new_r > best_r:
						mape_models.append(new_model)
						rs.append(new_r)

			# if no better mape models found, mape improvement no longer possible
			if mapes == [] or mape_models == []:
				gain_mape = False
			# else get the new best mape model
			else:
				best_mape_model, best_mape = self.getBest(mape_models, mapes, True)	

			# if no better r models found, r improvement no longer possible
			if rs == [] or r_models == []:
				gain_r = False
			# else get the new best r model
			else:
				best_r_model, best_r = self.getBest(r_models, rs, False)

		# get missing info for the best models
		m, r_for_mape = self.getPerformance([best_mape_model])
		mape_for_r, n = self.getPerformance([best_r_model])

		r_for_mape = r_for_mape[0]
		mape_for_r = mape_for_r[0]

		# append the best models
		final_models = [best_mape_model, best_r_model]		
		final_mape = [best_mape, mape_for_r]
		final_r = [r_for_mape, best_r]

		# return the best models
		return final_models, final_mape, final_r

	# gets the performance of models defined by model strings in the list models over epochs number of epochs
	def getPerformance(self, models, epochs=EPOCHS):
		# init
		mapes = []
		rs = []		

		# for every model string
		for model_str in models:
			# get target
			target = getTarget(model_str)			
			mapes_all = []
			rs_all = []
			# repeat epoch times
			for epoch in range(epochs):
				# get train x test data
				train, test = trainTestData(self.data)
				# get real target values
				real_y = np.asarray(test[target])
				# create and fit the model
				model = sm.formula.ols(formula = model_str, data = train)
				model = model.fit()
				# append r to the list of rs
				rs_all.append(model.rsquared)
				# get predicted target values
				pred_y = np.asarray(model.predict(test))	
				# calculate MAPE			
				mape = []
				for i in range(len(pred_y)):
					targ = real_y[i]
					pred = pred_y[i]
					if targ == 0:
						mape.append(abs(targ - pred))
					else:
						mape.append(abs(targ - pred)/targ)
				mapes_all.append(np.asarray(mape).mean())
			# get MAPE and r means over epochs
			mapes.append(np.asarray(mapes_all).mean())
			rs.append(np.asarray(rs_all).mean())

			# if this is a one variable model, generate its stats plot
			variables = getVars(model_str)
			if len(variables) == 1:
				model = sm.formula.ols(formula = model_str, data = self.data)
				model = model.fit()
				pred = model.predict(self.data)
				self.scPredPlot(variables[0], target, pred)
		# return the performance
		return mapes, rs