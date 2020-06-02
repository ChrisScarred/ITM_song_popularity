import pandas as pd
import statsmodels.api as sm
import numpy as np


class Analyses:
	def __init__(self, data, k):
		self.data = data
		# NOTE: (specify the models as formula strings)
		self.singeVarModels = []
		self.bestModels = []
		self.absoluteModels = []
		self.k = k

	# TODO
	def normalisedOrAbsolute(self):
		# compare the best models with normalised popularity to the same models with absolute popularity
		return 0

	# TODO
	def singleVarCorrelations(self):
		# get every single variable models with normalised popularity
		return 0

	# TODO
	def bestModelSearch(self):
		# obtain the best model(s) with normalised popularity
		return 0

	# gets performance of models (MSE) soecified by R-style strings over x epochs
	# example usage: getPerformance(["popularity_rel ~ tempo", "popularity_rel ~ danceability + valence"], 100, True)
	def getPerformance(self, models, epochs, doPrint):
		performances = []
		for epoch in range(epochs):
			print("Calculating epoch %d" % (epoch+1))
			kfolds = self.kfolds(self.k, True)

			test_indices = []
			if self.k > len(models):
				test_indices = np.random.choice([i for i in range(self.k)], len(models), replace=False)
			else:
				test_indices = np.random.choice([i for i in range(self.k)], len(models), replace=True)

			performance = []
			for i in range(len(models)):				
				test_data = kfolds[i]
				train_data = pd.DataFrame()

				for x in range(len(kfolds)):
					if x != i:
						dat = pd.DataFrame(kfolds[x])
						train_data = pd.concat([train_data, dat])

				model_str = models[i]
				model = sm.formula.ols(formula = model_str, data = train_data)
				model_fitted = model.fit()
				prediction = model_fitted.predict(test_data)
				mse = (np.square(test_data["popularity_rel"] - prediction)).mean(axis=None)
				performance.append(mse)

			performances.append(performance)

		performances = np.mean(performances, axis=0)

		if doPrint:
			for i in range(len(models)):
				print("Model %s has MSE of %.3f" % (models[i], performance[i]))

		return performance

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
