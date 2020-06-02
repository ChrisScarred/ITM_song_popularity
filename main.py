from controller import Controller

def main():
	# whether data should be obtained from spotify
	getData = False

	# whether data should be preprocessed
	doPreprocess = False

	# whether preprocessing or data pulling should occur 
	# automatically if one of the files is missing
	auto = True

	# perform singleVarCorrelations function in analyses
	svc = False

	# perform bestModelSearch function in analyses
	bms = False

	# perform normalisedOrAbsolute function in analyses
	noa = False

	controller = Controller(getData, doPreprocess, auto, svc, bms, noa)
	controller.performActions()
	

if __name__ == "__main__":
	main()