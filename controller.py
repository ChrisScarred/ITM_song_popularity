from preprocess import Preprocess
from dataGetter import DataGetter
from analyses import Analyses

import os.path
from os import path
import pandas as pd

ARTIST_URI = 'spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT'
NON_INCLUDE_ALBUMS = ["Blood at the Orpheum (Live)", "Blood"]
NON_INCLUDE_TRACKS = ["Interview (Bonus)"]

FOLDER = 'data'
NAME = FOLDER + '/itm_songs_database'
K = 10
EPOCHS = 10
R_THRESHOLD = 0.05
MAPE_THRESHOLD = 0.22


class Controller:
	def __init__(self, getData, doPreprocess, auto, svm, bms, noa, print_svm, print_bms, print_noa, brute_force, bf_pickle):
		self.getData = getData
		self.doPreprocess = doPreprocess
		self.auto = auto
		self.svm = svm
		self.bms = bms
		self.noa = noa
		self.print_svm = print_svm
		self.print_bms = print_bms
		self.print_noa = print_noa
		self.brute_force = brute_force
		self.bf_pickle = bf_pickle

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
			a = Analyses(data, K, EPOCHS, R_THRESHOLD, MAPE_THRESHOLD)
			if self.svm:
				a.singleVarModels(self.print_svm)
			if self.bms:
				a.bestModelSearch(self.print_bms, self.brute_force, self.bf_pickle)
			if self.noa:
				a.normalisedOrAbsolute(self.print_noa)		

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
