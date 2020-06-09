import statsmodels.api as sm
import numpy as np
from itertools import combinations
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
		# get vars
		x_vars = to_plot[0]
		y_vars = to_plot[1]
		colin_vars = to_plot[2]
		statplots = to_plot[3]
		# get combinations for colinearity analysis
		colin_pairs = combinations(colin_vars, 2)

		for x in x_vars:
			for y in y_vars:
				self.scplot(self.data[x], self.data[y])				
		for comb in colin_pairs:
			x = comb[0]
			y = comb[1]
			self.scplot(self.data[x], self.data[y])
		for model in statPlots:
			self.statPlots(model)

	def scplot(self, x, y):
		plt.figure()
		sns.scatterplot(x = x, y = y).set(title = ('Relationship between %s and %s' % (x, y)))
		name = 'plots/'+x+"_againts_"+y+".png"
		plt.savefig(name)

	def scPredPlot(self, x, y, prediction):
		sns.scatterplot(x = x, y = y).set(title = 'Relationship between %s and %s, including linear model' % )
		sns.lineplot(x = x, y = prediction, color = '#bd1d00')
		name = 'plots/'+x+"_againts_"+y+"with_prediction.png"
		plt.savefig(name)

	def statPlots(self, model_str):
		model = sm.formula.ols(formula=model_str, data=self.data)
		model_fitted = model.fit()

		intercept = model_fitted.params[0]
		variables = getVars(model_str)

		target = getTarget(model_str)

		predscores = 0
		resscores = 0

		for i in range(len(variables)):
			vname = variables[i]
			pred_score = intercept + model_fitted.params[i] * self.data[vname]
			res_score = np.abs(self.data[vname] - pred_score)
			predscores += pred_score
			resscores += res_score

		fig, axs = plt.subplots(2, 2, figsize=(15,15))

		sns.scatterplot(x = resscores, y = self.data[target], ax = axs[0,0]).set(
   			title = ('Absolute residuals against predicted values for model %s' % model_str), 
    		xlabel = 'Predicted scores', 
    		ylabel = 'Residuals')

		sns.distplot(resscores, bins = 15, ax = axs[0, 1]).set(
    		title = ('Histogram of residual scores for model %s' % model_str), 
    		xlabel = 'Residual scores', 
    		ylabel = 'Probability')

		scipy.stats.probplot(resscores, plot = axs[1, 0])
		axs[1, 0].get_lines()[0].set_markerfacecolor('#c5c5d6')
		axs[1, 0].get_lines()[0].set_markeredgecolor('#c5c5d6')

		sns.scatterplot(x = list(range(0, len(self.data[target]))), y = resscores, ax = axs[1,1]).set(
   			title = ('Residuals against order of collection for model %s' % model_str), 
  		  	xlabel = 'Order of collection', 
  		  	ylabel = 'Residuals')

		namestr = model_str.replace(" ", "_")
    	name = 'plots/statplot_'+namestr+".png"
		plt.savefig(name)

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

	def printCustom(self, models):
		mapes, rs = self.getPerformance(models)
		self.printModels(models, mapes, rs, string="Stats for custom models:")
		return models

	
	def singleVarModels(self):	
		targets, variables = self.getTargVars()
		
		models = []

		for target in targets:			
			for var in variables:
				model = target + " ~ " + var
				models.append(model)
		
		mape, r = self.getPerformance(models)
		
		return models, mape, r

	def bestSingleVars(self, models, mapes, rs):
		suff_models, mape_new, r_new = self.getSufficient(models, mapes, rs)
		return suff_models, mape_new, r_new
	
	def getSufficient(self, i_models, i_mape, i_r):
		models = []
		mapes = []
		rs = []

		zipped = zip(i_models, i_mape, i_r)

		for (model, mape, r) in zipped:
			if (mape < MAPE_THRESHOLD) or (r > R_THRESHOLD):
				models.append(model)
				mapes.append(mape)
				rs.append(r)
		
		if len(models) == 0:
			logPrintIt(self.log, "No sufficiently well-performing model with single variable found.")

		return models, mapes, rs


	def printModel(self, model, mape, r):
		logPrintIt(self.log, ("Model '%s' with MAPE of %.3f and r^2 of %.3f" % (model, mape, r)))

	def printModels(self, models, mape, r, string="Queried models:"):
		s = "----------------------------------"
		string = "\n" + s + "\n" + string + "\n" + s
		logPrintIt(self.log, string)

		for i in range(len(models)):
			self.printModel(models[i], mape[i], r[i])


	def modelGetter(self, models_file, mapes_file, rs_file, func, string, fail):

		models, r1 = pickleLoader(models_file, self.log)
		mapes, r2 = pickleLoader(mapes_file, self.log)
		rs, r3 = pickleLoader(rs_file, self.log)

		if not r1 or not r2 or not r3:
			models, mapes, rs = func

			if len(models) > 0:
				pickleWriter(models, models_file, self.log)
				pickleWriter(mapes, mapes_file, self.log)
				pickleWriter(rs, rs_file, self.log)

		if len(models) > 0:
			self.printModels(models, mapes, rs, string=string)
			
		else:
			logPrintIt(self.log, fail)

		return models, mapes, rs

	
	def bestModelSearch(self):
		models, mapes, rs = self.modelGetter(SVM_MODELFILE, SVM_MAPEFILE, SVM_RFILE, self.singleVarModels(), 
			"All single variable models:", "No single variable model found.")

		models2, mapes2, rs2 = self.modelGetter(BSVM_MODELFILE, BSVM_MAPEFILE, BSVM_RFILE, self.bestSingleVars(models, mapes, rs), 
			"Best single variable models:", "No well-performing single variable model found.")

		models3, mapes3, rs3 = self.modelGetter(FS_MODELFILE, FS_MAPEFILE, FS_RFILE, self.forward_selection(models, mapes, rs), 
			"Best models obtained via forward selection:", "No well-performing model found via forward selection.")
		
		models4, mapes4, rs4 = self.modelGetter(BS_MODELFILE, BS_MAPEFILE, BS_RFILE, self.backward_selection(), 
			"Best models obtained via backward selection:", "No well-performing model found via backward selection.")
		
		models.extend(models2)
		models.extend(models3)
		models.extend(models4)

		return models 

	def getBest(self, models, criterion, lower):			
		zipped = zip(models, criterion)		
		res = sorted(zipped, key=lambda x: x[1])

		if lower:
			return res[0][0], res[0][1]

		return res[-1][0], res[-1][1]

	def getTargVars(self):
		no_includes = ["name", "id"]
		targets = ["popularity_rel", "popularity_abs"]
		variables = [c for c in self.data.columns if c not in targets and c not in no_includes]
		return targets, variables


	def forward_selection(self, singleVarModels, mape, r):					
		best_mape_model, best_mape = self.getBest(singleVarModels, mape, True)
		best_r_model, best_r = self.getBest(singleVarModels, r, False)

		targets, variables = self.getTargVars()

		gain_mape = True
		gain_r = True
		while gain_mape or gain_r:
			mapes = []
			rs = []
			mape_models = []
			r_models = []

			for var in variables:
				if var not in best_mape_model and gain_mape:
					model = best_mape_model + " + " + var

					mape, r = self.getPerformance([model])
					mape = mape[0]
					if mape < best_mape:
						mape_models.append(model)
						mapes.append(mape)

				
				if var not in best_r_model and gain_r:
					model = best_r_model + " + " + var

					mape, r = self.getPerformance([model])
					r = r[0]
					if r > best_r:
						r_models.append(model)
						rs.append(r)

			if mapes == [] or mape_models == []:
				gain_mape = False
			else:
				best_mape_model, best_mape = self.getBest(mape_models, mapes, True)				
				
			if rs == [] or r_models == []:
				gain_r = False
			else:
				best_r_model, best_r = self.getBest(r_models, rs, False)				
				

		m, r_for_mape = self.getPerformance([best_mape_model])
		mape_for_r, n = self.getPerformance([best_r_model])

		r_for_mape = r_for_mape[0]
		mape_for_r = mape_for_r[0]

		final_models = [best_mape_model, best_r_model]		
		final_mape = [best_mape, mape_for_r]
		final_r = [r_for_mape, best_r]

		return final_models, final_mape, final_r

	def getBackString(self, model, var):
		str_new = model.replace(var, "")
		str_new = str_new.replace("~  + ", "~ ")
		str_new = str_new.replace("+  + ", "+ ")
		str_new = str_new.strip()

		if str_new.endswith("+"):
			str_new = str_new[:-1].strip()

		if not(str_new.split(" ~ ")[-1] == ""):		
			model = str_new					
			mape, r = self.getPerformance([model])
			mape = mape[0]
			r = r[0]
			return model, mape, r 

		else:
			return False, False, False

	def backward_selection(self):
		targets, variables = self.getTargVars()

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

		mape, r = self.getPerformance(models)

		best_mape_model, best_mape = self.getBest(models, mape, True)
		best_r_model, best_r = self.getBest(models, r, False)

		gain_mape = True
		gain_r = True
		while gain_mape or gain_r:
			mapes = []
			rs = []
			mape_models = []
			r_models = []

			for var in variables:
				targ = getTarget(best_mape_model)
				if var in best_mape_model and gain_mape:
					new_model, new_mape, new_r = self.getBackString(best_mape_model, var)
					if not new_model:
						pass
					elif new_mape < best_mape:
						mape_models.append(new_model)
						mapes.append(new_mape)

				targ = getTarget(best_r_model)
				if var in best_r_model and gain_r:
					new_model, new_mape, new_r = self.getBackString(best_r_model, var)
					if not new_model:
						pass
					elif new_r > best_r:
						mape_models.append(new_model)
						rs.append(new_r)

			if mapes == [] or mape_models == []:
				gain_mape = False
			else:
				best_mape_model, best_mape = self.getBest(mape_models, mapes, True)		

			if rs == [] or r_models == []:
				gain_r = False
			else:
				best_r_model, best_r = self.getBest(r_models, rs, False)

		m, r_for_mape = self.getPerformance([best_mape_model])
		mape_for_r, n = self.getPerformance([best_r_model])

		r_for_mape = r_for_mape[0]
		mape_for_r = mape_for_r[0]

		final_models = [best_mape_model, best_r_model]		
		final_mape = [best_mape, mape_for_r]
		final_r = [r_for_mape, best_r]

		return final_models, final_mape, final_r


	def getPerformance(self, models, epochs=EPOCHS):
		mapes = []
		rs = []		

		for model_str in models:
			target = getTarget(model_str)			
			mapes_all = []
			rs_all = []
			for epoch in range(epochs):
				train, test = trainTestData(self.data)				
				real_y = np.asarray(test[target])
				model = sm.formula.ols(formula = model_str, data = train)
				model = model.fit()
				rs_all.append(model.rsquared)
				pred_y = np.asarray(model.predict(test))

				if epoch == epochs-1:
					self.scPredPlot(test, real_y, pred_y)
				
				mape = []
				for i in range(len(pred_y)):
					targ = real_y[i]
					pred = pred_y[i]
					if targ == 0:
						mape.append(abs(targ - pred))
					else:
						mape.append(abs(targ - pred)/targ)
				mapes_all.append(np.asarray(mape).mean())
			mapes.append(np.asarray(mapes_all).mean())
			rs.append(np.asarray(rs_all).mean())

		return mapes, rs