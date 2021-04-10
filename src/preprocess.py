import pandas as pd

'''
simple data preprocessor
reads data
returns normalised data with target variable both in relative terms and absolute
'''
class Preprocess:
	# initialisation
	def __init__(self, name):
		self.data = pd.read_csv(name+".csv")
		self.name = name
		
	def preprocess(self):
		print("preprocessing "+self.name+".csv")
		# rename for clearer reference
		self.data = self.data.rename(columns={"duration_ms" : "duration", "popularity": "popularity_abs"})
		# copy target to a new column
		self.data["popularity_rel"] = self.data["popularity_abs"]
		# normalise target in new column
		self.normalise("popularity_rel")
		# normalise other non-normalised variables
		self.normalise("complexity")
		self.normalise("duration")
		self.normalise("loudness")
		# drop automatically added unnamed column
		self.data = self.data.drop(["Unnamed: 0"], axis=1)
		# save
		print("saving to "+self.name+"_preprocessed.csv")
		self.data.to_csv(self.name+"_preprocessed.csv")
		print("done")
		return self.data

	# trivial normalising function
	def normalise(self, col):
		max_val = max(self.data[col])
		min_val = min(self.data[col])
		diff = max_val - min_val
		self.data[col] = self.data[col].apply(lambda x: (x-min_val)/diff)