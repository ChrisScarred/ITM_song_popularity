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

TARGET = ""

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

RTBF = 0.10
MTBF = 0.10
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
			print(models)
			print(string)
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

	
	def bestModelSearch(self, brute_force):
		models, mapes, rs = self.modelGetter(SVM_MODELFILE, SVM_MAPEFILE, SVM_RFILE, self.singleVarModels(), 
			"All single variable models:", "No single variable model found.")

		models2, mapes2, rs2 = self.modelGetter(BSVM_MODELFILE, BSVM_MAPEFILE, BSVM_RFILE, self.bestSingleVars(models, mapes, rs), 
			"Best single variable models:", "No well-performing single variable model found.")

		models3, mapes3, rs3 = self.modelGetter(FS_MODELFILE, FS_MAPEFILE, FS_RFILE, self.forward_selection(models, mapes, rs), 
			"Best models obtained via forward selection:", "No well-performing model found via forward selection.")

		models4, mapes4, rs4 = self.modelGetter(BS_MODELFILE, BS_MAPEFILE, BS_RFILE, self.backward_selection(), 
			"Best models obtained via backward selection:", "No well-performing model found via backward selection.")
		models5, mapes5, rs5 = [], [], []
		if brute_force:
			models5, mapes5, rs5 = self.doBruteForce()

		models.append(models2)
		models.append(models3)
		models.append(models4)
		models.append(models5)

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

	def doBruteForce(self):
		models, r1 = pickleLoader(BF_MODELFILE, self.log)
		mapes, r2 = pickleLoader(BF_MAPEFILE, self.log)
		rs, r3 = pickleLoader(BF_RFILE, self.log)
		means, r4 = pickleLoader(MEAN_FILE, self.log)

		if not r1 or not r2 or not r3 or not r4:
			models, mapes, rs, mean_mapes_abs, mean_mapes_rel, mean_r_abs, mean_r_rel = self.brute_force()
			means = [mean_mapes_abs, mean_mapes_rel, mean_r_abs, mean_r_rel]

		if len(models) > 0:
			pickleWriter(models, BF_MODELFILE, self.log)
			pickleWriter(mapes, BF_MAPEFILE, self.log)
			pickleWriter(rs, BF_RFILE, self.log)
			pickleWriter(means, MEAN_FILE, self.log)

			for m in models:
				if m not in self.bestModels:
					self.bestModels.append(m)

			targets = ["popularity_rel", "popularity_abs"]
			string = ("\n----------------------------------\nBased on comparison of all possible models, the average MAPE for '%s' is %.3f and for '%s' %.3f while the average r^2 for '%s' is %.3f and for '%s' %.3f"
				% (targets[1], means[0], targets[0], means[1], targets[1], means[2], targets[0], means[3]))
			logPrintIt(self.log, string)

		else:
			logPrintIt(self.log, "No well-performing model found via brute force search.")
		return models, mapes, rs

	def brute_force(self):
		models = []
		targets, variables = self.getTargVars()	
		i = len(variables)-1
		combs = [[variables]]

		while i > 1:
			comb = [j for j in combinations(variables, i)]
			combs.append(comb)
			i -= 1

		combs = [item for sublist in combs for item in sublist]

		for target in targets:
			mdls = []			
			for vrs in combs:
				model_str = target + " ~ "
				for i in range(len(vrs)):
					if i == len(vrs)-1:
						model_str += vrs[i]
					else:
						model_str += vrs[i]
						model_str += " + "				
				mdls.append(model_str)
			models.append((target, mdls))

		bestModels = []
		best_mapes = []
		best_rs = []

		all_mapes_abs = []
		all_mapes_rel = []
		all_r_abs = []
		all_r_rel = []

		# number of chunks and processes equal to cpu count
		processes = int(os.getenv('CPU_COUNT', cpu_count()))

		for i in range(len(models)):
			target = getTarget(models[i][0])
			global TARGET
			TARGET = target

			chunk = chunks(models[i], processes)			

			with Pool(processes=processes) as pool:
				a = pool.map(self.brute_force_parallel, chunk)

				print(a)
				
				bestModels.append(mdls1)
				best_mapes.append(mapes_i1)
				best_rs.append(rs_i1)

				all_mapes_abs.extend(all_mapes_abs1)
				all_mapes_rel.extend(all_mapes_rel1)
				all_r_abs.extend(all_r_abs1)
				all_r_rel.extend(all_r_rel1)

		mean_mapes_abs = np.mean(all_mapes_abs)
		mean_mapes_rel = np.mean(all_mapes_rel)
		mean_r_abs = np.mean(all_r_abs)
		mean_r_rel = np.mean(all_r_rel)

		self.printModels(bestModels, best_mapes, best_rs, string="The best models obtained by brute force:")

		return bestModels, best_mapes, best_rs, mean_mapes_abs, mean_mapes_rel, mean_r_abs, mean_r_rel

	def brute_force_parallel(self, chunk):
		mdls = []
		mapes_i = []
		rs_i = []
		all_mapes_abs = []
		all_mapes_rel = []
		all_r_abs = []
		all_r_rel = []

		for model in chunk[0]:
			train, test = trainTestData(self.data)
			mdl = sm.formula.ols(formula = model, data = train)
			model_fitted = mdl.fit()
			r = model_fitted.rsquared
			prediction = list(model_fitted.predict(test))
			target_vals = list(test[TARGET])

			mape = []
			for i in range(len(prediction)):
				targ = target_vals[i]
				pred = prediction[i]
				if targ == 0:
					mape.append(abs(targ - pred))
				else:
					mape.append(abs(targ - pred)/targ)
			mape = np.mean(mape)

			if "_abs" in model:
				all_mapes_abs.append(mape)
				all_r_abs.append(r)
			else:
				all_mapes_rel.append(mape)
				all_r_rel.append(r)

			if r > RTBF and mape < MTBF:
				mdls.append(model)
				mapes_i.append(mape)
				rs_i.append(r)

		return mdls, mapes_i, rs_i, all_mapes_abs, all_mapes_rel, all_r_abs, all_r_rel


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

	