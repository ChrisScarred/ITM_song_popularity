from preprocess import Preprocess
from dataGetter import DataGetter
from analyses import Analyses

import os.path
from os import path
import pandas as pd

ARTIST_URI = 'spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT'
NON_INCLUDE_ALBUMS = ["Blood at the Orpheum (Live)", "Blood"]
NON_INCLUDE_TRACKS = ["Interview (Bonus)"]

NAME = 'itm_songs_database'
K = 10

class Controller:
	def __init__(self, getData, doPreprocess, auto, svc, bms, noa):
		self.getData = getData
		self.doPreprocess = doPreprocess
		self.auto = auto
		self.svc = svc
		self.bms = bms
		self.noa = noa

	def performActions(self):
		data = []

		if self.getData:
			data = obtainData()

		if self.doPreprocess:		
			if(path.exists(NAME+".csv")):
				data = self.performPreprocess()
			else:
				print("File "+NAME+".csv does not exist, preprocessing not possible before the file is created.")
				if self.auto:
					print("Running DataGetter now")
					self.obtainData()
					data = self.performPreprocess()

		else:		
			if(path.exists(NAME+"_preprocessed.csv")):
				data = pd.read_csv(NAME+"_preprocessed.csv")
				data = data.drop(["Unnamed: 0"], axis=1)
			else:
				print("File "+NAME+"_preprocessed.csv does not exist, no analysis possible before the file is created.")

				if self.auto:
					print("Running preprocesser now")
					data = self.performPreprocess()


		if path.exists(NAME+"_preprocessed.csv"):
			a = Analyses(data, K)
			if self.svc:
				a.singleVarCorrelations()
			if self.bms:
				a.bestModelSearch()
			if self.noa:
				a.normalisedOrAbsolute()		

		else:
			print("File does not exist, not performing analysis.")

	def obtainData(self):
		dg = DataGetter(ARTIST_URI, NON_INCLUDE_ALBUMS, NON_INCLUDE_TRACKS, NAME)
		data = dg.getData()

	def performPreprocess(self):
		data = []
		if(path.exists(NAME+".csv")):
			pr = Preprocess(NAME)
			data = pr.preprocess()
		else:
			print("File "+NAME+".csv does not exist, preprocessing not possible before the file is created.")
			if self.auto:
				print("Running DataGetter now")
				obtainData()
				pr = Preprocess(NAME)
				data = pr.preprocess()
		return data
