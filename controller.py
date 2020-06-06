from preprocess import Preprocess
from dataGetter import DataGetter
from analyses import Analyses
from utils import logPrintIt

import os.path
from os import path
import pandas as pd
import logging


ARTIST_URI = 'spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT'
NON_INCLUDE_ALBUMS = ["Blood at the Orpheum (Live)", "Blood"]
NON_INCLUDE_TRACKS = ["Interview (Bonus)"]

FOLDER = 'database'
NAME = FOLDER + '/itm_songs_database'

class Controller:
	def __init__(self, getData, doPreprocess, auto, brute_force, log, custom, full_sum):
		self.getData = getData
		self.doPreprocess = doPreprocess
		self.auto = auto

		self.brute_force = brute_force

		self.log = log
		self.custom = custom
		self.full_sum = full_sum
		
		if log:
			logging.basicConfig(format='%(message)s', filename='results.log',level=logging.INFO)
			logging.info('------------------------------------------------------------')		


	def autoDataGet(self):
		data = []

		if not (path.exists(NAME+".csv")):
			data = self.obtainData()

		if not (path.exists(NAME+"_preprocessed.csv")):
			data = self.performPreprocess()
		data = pd.read_csv(NAME+"_preprocessed.csv")
		data = data.drop(["Unnamed: 0"], axis=1)

		return data


	def manualDataGet(self):
		data = []
		if self.getData:
			data = self.obtainData()
		if self.doPreprocess:		
			if(path.exists(NAME+".csv")):
				data = self.performPreprocess()
			else:
				self.logPrintIt(self.log, self.log, "File "+NAME+".csv does not exist, preprocessing not possible before the file is created.")
				return data
				
		elif (path.exists(NAME+"_preprocessed.csv")):
			data = pd.read_csv(NAME+"_preprocessed.csv")
			data = data.drop(["Unnamed: 0"], axis=1)

		else:
			self.logPrintIt(self.log, self.log, "File "+NAME+"_preprocessed.csv does not exist, no analysis possible before the file is created.")
		
		return data	

	def composeData(self):
		data = []
		if self.auto:
			data = self.autoDataGet()
		else:
			data = self.manualDataGet()
		return data

	def analyse(self, data):
		models = []

		a = Analyses(data, self.log)
		
		models2 = a.bestModelSearch(self.brute_force)

		models3 = []
		if self.custom != []:
			models3 = a.printCustom(self.custom)

		models.append(models2)
		models.append(models3)

		if self.full_sum:
			a.printSummaries(models)

	def performActions(self):
		data = self.composeData()
		self.analyse(data)

	def obtainData(self):
		dg = DataGetter(ARTIST_URI, NON_INCLUDE_ALBUMS, NON_INCLUDE_TRACKS, NAME)
		data = dg.getData()

	def performPreprocess(self):
		pr = Preprocess(NAME)
		data = pr.preprocess()
		return data
