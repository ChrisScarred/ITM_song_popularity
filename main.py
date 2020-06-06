from controller import Controller

def main():
	# whether data should be obtained from spotify
	getData = False

	# whether data should be preprocessed
	doPreprocess = False

	# whether preprocessing or data pulling should occur 
	# automatically if one of the files is missing
	auto = True

	# get all single var models
	svm = False
	print_svm = True

	# get best models
	bms = True
	print_bms = True

	# get results on abs vs rel
	noa = False
	print_noa = True

	# whether to do brute force analysis
	# WARNING: over 64k models are evaluated, it takes some time...
	brute_force = True
	# attempts to read bf models from file before creating them
	bf_pickle = True

	controller = Controller(getData, doPreprocess, auto, svm, bms, noa, print_svm, print_bms, print_noa, brute_force, bf_pickle)
	controller.performActions()
	

if __name__ == "__main__":
	main()