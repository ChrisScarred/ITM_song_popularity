from controller import Controller

def main():
	# whether data should be obtained from spotify
	getData = False

	# whether data should be preprocessed
	doPreprocess = False

	'''
	whether preprocessing or data pulling should occur 
	automatically if one of the files is missing
	'''
	auto = True

	# instead of printing results, write them into results.log
	log = True

	# custom models to print
	custom = []

	# creates summary files for all well-performing and queried models
	full_sum = True

	controller = Controller(getData, doPreprocess, auto, log, custom, full_sum)
	controller.performActions()
	

if __name__ == "__main__":
	main()