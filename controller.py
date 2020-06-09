from preprocess import Preprocess
from dataGetter import DataGetter
from analyses import Analyses
from utils import logPrintIt

import os.path
from os import path
import pandas as pd
import logging

# artist id obtained via spotify
ARTIST_URI = 'spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT'

# tracks and albums that should not be included because they are either duplicates or live versions
NON_INCLUDE_ALBUMS = ["Blood at the Orpheum (Live)", "Blood"]
NON_INCLUDE_TRACKS = ["Interview (Bonus)"]

# database folder and path
FOLDER = 'database'
NAME = FOLDER + '/itm_songs_database'

'''
Controls the actions of all other submodules based on the input
from main.py

getData [boolean] - whether to pull data
doPreprocess [boolean] - whether to preprocess data 
auto [boolean] - whether to do the above automatically if missing
log [boolean] - whether to log to file [true] or console [false]
custom [list of strings] - models to evaluate regardless of what analysis returns
full_sum [boolean] - whther to save summaries into files
to_plot [tuple of lists of strings] - models/vars to print, explained in main.py
'''
class Controller:
	def __init__(self, getData, doPreprocess, auto, log, custom, full_sum, to_plot):
		# initialisation
		self.getData = getData
		self.doPreprocess = doPreprocess
		self.auto = auto

		self.log = log
		self.custom = custom
		self.full_sum = full_sum

		self.to_plot = to_plot
		
		# initialising the logger
		if log:
			logging.basicConfig(format='%(message)s', filename='results.log',level=logging.INFO)
			logging.info('-'*80)		

	# obtains data automatically - if any missing, gets it
	def autoDataGet(self):
		data = []

		# pulls raw data from spotify if missing
		if not (path.exists(NAME+".csv")):
			data = self.obtainData()

		# preprocesses the raw data if missing
		if not (path.exists(NAME+"_preprocessed.csv")):
			data = self.performPreprocess()

		# reads csv info df
		data = pd.read_csv(NAME+"_preprocessed.csv")
		# drops automatically added unnamed column
		data = data.drop(["Unnamed: 0"], axis=1)

		return data

	# obtains data if indicated by settings, if data missing, halts analyses
	def manualDataGet(self):
		data = []
		# obtains raw data if indicated it should
		if self.getData:
			data = self.obtainData()

		# preprocesses raw data if indicated it should, halts if data missing
		if self.doPreprocess:		
			if(path.exists(NAME+".csv")):
				data = self.performPreprocess()
			else:
				self.logPrintIt(self.log, self.log, "File "+NAME+".csv does not exist, preprocessing not possible before the file is created.")
				return data

		# loads preprocessed data	
		elif (path.exists(NAME+"_preprocessed.csv")):
			data = pd.read_csv(NAME+"_preprocessed.csv")
			data = data.drop(["Unnamed: 0"], axis=1)
		# or halts if preprocessed data does not exist
		else:
			self.logPrintIt(self.log, self.log, "File "+NAME+"_preprocessed.csv does not exist, no analysis possible before the file is created.")
		
		return data	

	# runs one of the data obtaining scripts
	def composeData(self):
		data = []
		if self.auto:
			data = self.autoDataGet()
		else:
			data = self.manualDataGet()
		return data

	# runs analyses scripts
	def analyse(self, data):
		# initialise
		a = Analyses(data, self.log)
		
		# runs the main analysis function
		models = a.bestModelSearch()

		# prints or logs custom model data if any
		if self.custom != []:
			models2 = a.printCustom(self.custom)
			models.extend(models2)

		# saves summaries if indicated
		if self.full_sum:
			a.printSummaries(models)

		# runs plot generation if to_plot not empty
		if self.to_plot != []:
			a.make_plots(self.to_plot)

	# main function of this module; gets data and anmalises it
	def performActions(self):
		data = self.composeData()
		self.analyse(data)

	# main function for data obtaining
	def obtainData(self):
		dg = DataGetter(ARTIST_URI, NON_INCLUDE_ALBUMS, NON_INCLUDE_TRACKS, NAME)
		data = dg.getData()
		return data

	# main function for data preprocessing
	def performPreprocess(self):
		pr = Preprocess(NAME)
		data = pr.preprocess()
		return data
