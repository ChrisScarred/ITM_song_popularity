import pandas as pd
import statsmodels.api as sm
import numpy as np

from itertools import combinations
from random import randint
from multiprocessing import Pool, cpu_count
from utils import *

import os
import pickle
import logging

from pathlib import Path

root = Path(".")

BF_MODELFILE = root / "data" / 'bf_models.pickle'
BF_MAPEFILE = root / "data" / 'bf_mape.pickle'
BF_RFILE = root / "data" / 'bf_r.pickle'

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

MEAN_FILE = root / "data" / 'mean.pickle'


EPOCHS = 10
R_THRESHOLD = 0.05
MAPE_THRESHOLD = 0.20

SUM_FOLDER = 'summaries/'

class Analyses:
	def __init__(self, data, log):
		self.data = data
		self.log = log


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