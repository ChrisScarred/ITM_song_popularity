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
	print_bms = False

	# get results on abs vs rel
	noa = False
	print_noa = True

	# whether to do brute force analysis
	# WARNING: over 64k models are evaluated, it takes some time...
	brute_force = True

	controller = Controller(getData, doPreprocess, auto, svm, bms, noa, print_svm, print_bms, print_noa, brute_force)
	controller.performActions()
	

if __name__ == "__main__":
	main()