import pandas as pd

class Preprocess:
	def __init__(self, name):
		self.data = pd.read_csv(name+".csv")
		self.name = name
		
	def preprocess(self):
		print("preprocessing "+self.name+".csv")
		self.data = self.data.rename(columns={"duration_ms" : "duration", "popularity": "popularity_abs"})
		self.data["popularity_rel"] = self.data["popularity_abs"]
		self.normalise("popularity_rel")
		self.normalise("complexity")
		self.normalise("duration")
		self.normalise("loudness")
		self.data = self.data.drop(["Unnamed: 0"], axis=1)
		print("saving to "+self.name+"_preprocessed.csv")
		self.data.to_csv(self.name+"_preprocessed.csv")
		print("done")
		return self.data

	def normalise(self, col):
		max_val = max(self.data[col])
		min_val = min(self.data[col])
		diff = max_val - min_val
		self.data[col] = self.data[col].apply(lambda x: (x-min_val)/diff)