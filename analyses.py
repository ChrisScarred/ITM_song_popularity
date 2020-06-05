import pandas as pd
import statsmodels.api as sm
import numpy as np
from itertools import combinations
from random import randint
from multiprocessing import Pool, cpu_count
import os


TARGET = 0

class Analyses:
	def __init__(self, data, k, epochs, r_threshold, mape_threshold):
		self.data = data
		# NOTE: (specify the models as formula strings)
		self.bestModels = []
		self.k = k
		self.epochs = epochs
		self.r_threshold = r_threshold
		self.mape_threshold = mape_threshold

	# TODO
	def normalisedOrAbsolute(self):
		# compare the best models with normalised popularity to the same models with absolute popularity
		return 0

	# TODO
	def singleVarModels(self, print_it):
		singleVarModels = []
		no_includes = ["name", "id"]
		targets = ["popularity_rel", "popularity_abs"]
		cols = [c for c in self.data.columns if c not in targets and c not in no_includes]		
		
		for target in targets:
			models = []
			for col in cols:
				model = target + " ~ " + col
				models.append(model)
			singleVarModels.append((target, models))

		mape, r = self.getPerformance(singleVarModels, print_it)
		return singleVarModels, mape, r

	def bestSingleVars(self, print_it):
		models, mape, r = self.singleVarModels(print_it)
		suff_models, mape_new, r_new = self.getSufficient(mape, r, models, print_it)
		return suff_models, mape_new, r_new

	
	def getSufficient(self, mape, r, models, print_it):
		suff_models = []
		mape_new = []
		r_new = []

		for i in range(len(mape)):
			mape_i = []
			r_i = []

			for j in range(len(mape[i])):
				if (mape[i][j] < self.mape_threshold) or (r[i][j] > self.r_threshold):
					mdl = models[i][1][j]
					target = models[i][0]
					model = (target, [mdl])
					mape_i.append(mape[i][j])
					r_i.append(r[i][j])

					if model not in self.bestModels:
						self.bestModels.append(model)

					suff_models.append(model)

			mape_new.append(mape_i)
			r_new.append(r_i)
		
		if len(suff_models) == 0:
			print("No sufficiently well-performing model with single variable found.")
		else:
			print("%d sufficiently well-performing models with single variable were found." % len(suff_models))

			if print_it:
				self.printModels(mape_new, r_new, self.bestModels, string="All sufficiently well-performing models found so far are:")

		return suff_models, mape_new, r_new


	def printModel(self, model, mape, r):
		print("Model %s with MAPE of %.3f and r^2 of %.3f" % (model, mape, r))

	def printModels(self, models, mape, r, string="Queried models:"):
		print(string)

		pure_models = [i[1] for i in models]
		pure_models = [item for sublist in pure_models for item in sublist]
		print_perfs = [item for sublist in mape for item in sublist]
		print_r = [item for sublist in r for item in sublist]

		for i in range(len(pure_models)):			
			self.printModel(pure_models[i], print_perfs[i], print_r[i])

	# TODO
	def bestModelSearch(self, print_it, brute_force):		
		'''
		s_models, s_mape, s_r = self.bestSingleVars(False)
		if len(s_models) > 0:
			if print_it:
				self.printModels(s_models, s_mape, s_r, string="Best single variable models:")
			else:
				print("%d well-performing single variable models found." % len(bs_models))			
		else:
			print("No well-performing single variable model found.")
		'''

		f_models, f_mape, f_r  = self.forward_selection()
		if len(f_models) > 0:
			if print_it:
				self.printModels(f_models, f_mape, f_r, string="Best models obtained via forward selection:")
			else:
				print("%d well-performing models found via forward selection." % len(bs_models))
		else:
			print("No well-performing model found via forward selection.")

		bs_models, bs_mape, bs_r  = self.backward_selection()
		if len(bs_models) > 0:
			if print_it:
				self.printModels(bs_models, bs_mape, bs_r, string="Best models obtained via backward selection:")
			else:
				print("%d well-performing models found via backward selection." % len(bs_models))
		else:
			print("No well-performing model found via backward selection.")

		if brute_force:
			bf_models, bf_mape, bf_r  = self.brute_force()
			if len(bf_models) > 0:
				if print_it:
					self.printModels(bf_models, bf_mape, bf_r, string="Best models obtained via brute force search:")
				else:
					print("%d well-performing models found via brute force search." % len(bf_models))
			else:
				print("No well-performing model found via brute force search.")

	def forward_selection(self):
		return [],[],[]

	def backward_selection(self):
		return [],[],[]

	def brute_force(self):
		models = []
		no_includes = ["name", "id"]
		targets = ["popularity_rel", "popularity_abs"]
		variables = tuple([x for x in self.data.columns if x not in no_includes and x not in targets])	
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

		best_models = []
		best_mapes = []
		best_rs = []

		# number of chunks and processes equal to cpu count
		processes = int(os.getenv('CPU_COUNT', cpu_count()))

		for i in range(len(models)):
			target = models[i][0]
			global TARGET
			TARGET = target

			chunk = self.chunks(models[i][1],processes)			

			print("parallel processing of brute force obtained models started")
			with Pool(processes=processes) as pool:
				results = pool.map(self.brute_force_parallel, chunk)
				print(results)
				best_models.append((target, results[0][0]))
				best_mapes.append(results[0][1])
				best_rs.append(results[0][2])

			print("parallel processing of brute force obtained models ended")

		return best_models,best_mapes,best_rs

	def brute_force_parallel(self, chunk):
		mdls = []
		mapes_i = []
		rs_i = []

		for model in chunk[0]:
			train, test = self.trainTestData()
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

			if r > self.r_threshold or mape < self.mape_threshold:
				mdls.append(model)
				mapes_i.append(mape)
				rs_i.append(r)

		return [[mdls], [mapes_i], [rs_i]]


	# gets performance of models (MAPE) soecified by R-style strings over x epochs
	# example usage: getPerformance(["popularity_rel ~ tempo", "popularity_rel ~ danceability + valence"], 100, True)
	def getPerformance(self, models, doPrint):
		performances = []
		r = []

		pure_models = [i[1] for i in models]
		pure_models = [item for sublist in pure_models for item in sublist]
			
		no_of_models = len(pure_models)
		no_of_targets = len(models)
			
		for epoch in range(self.epochs):
			print("Calculating epoch %d" % (epoch+1))
			kfolds = self.kfolds(self.k, True)
	
			test_indices = []
			if self.k > no_of_models:
				test_indices = np.random.choice([i for i in range(self.k)], no_of_models, replace=False)				
			else:
				test_indices = np.random.choice([i for i in range(self.k)], no_of_models, replace=True)

			perf_cats = []
			r_cats = []

			for i in range(len(models)):
				target = models[i][0]
				perf_cat = []
				r_cat = []

				for model_str in models[i][1]:

					test_index = test_indices[i]		
					test_data = kfolds[test_index]

					train_data = pd.DataFrame()

					for x in range(len(kfolds)):
						if x != test_index:
							dat = pd.DataFrame(kfolds[x])
							train_data = pd.concat([train_data, dat])
					
					model = sm.formula.ols(formula = model_str, data = train_data)
					model_fitted = model.fit()
					r_cat.append(model_fitted.rsquared)

					prediction = list(model_fitted.predict(test_data))
					target_vals = list(test_data[target])

					mape = []
					for i in range(len(prediction)):
						targ = target_vals[i]
						pred = prediction[i]
						if targ == 0:
							mape.append(abs(targ - pred))
						else:
							mape.append(abs(targ - pred)/targ)
					mape = np.mean(mape)				
					perf_cat.append(mape)

				r_cats.append(r_cat)
				perf_cats.append(perf_cat)
			r.append(r_cats)
			performances.append(perf_cats)

		performances = np.mean(performances, axis=0)
		r = np.mean(r, axis=0)

		if doPrint:
			self.printModels(performances, r, models, string="Performances of queried models:")

		return performances, r

	# creates k partitions of data, if shuffle, shuffles the order of data
	def kfolds(self, k, shuffle):
		data = self.data
		if shuffle:
			data = self.data.sample(frac=1)

		folds = []
		length = len(data)
		addition = 0
		size = 0

		if length % k == 0:
			size = int(length / k)
		else:
			addition = length % k
			size = int((length - addition) / k)

		for i in range(k):
			if i == k-1:
				folds.append(data[(i*size):])
			else:
				folds.append(data[(i*size):((i+1)*size)])

		return folds

	def trainTestData(self):
		folds = self.kfolds(self.k, True)
		i = randint(0, len(folds)-1)
		train_data = pd.DataFrame()
		for x in range(len(folds)):
			if x != i:
				dat = pd.DataFrame(folds[x])
				train_data = pd.concat([train_data, dat])
		return train_data, folds[i]


	def chunks(self, data, processes):
		folds = []
		length = len(data)
		addition = 0
		size = 0

		if length % processes == 0:
			size = int(length / processes)
		else:
			addition = length % processes
			size = int((length - addition) / processes)

		for i in range(processes):
			if i == processes-1:				
				folds.append(data[(i*size):])
			else:
				folds.append(data[(i*size):((i+1)*size)])

		yield folds