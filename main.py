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
	# this is the best forward/backward selection model based on r
	custom = [""]

	# plots these vars against 
	to_plot_x = ['key', 'mode', 'time_signature', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness',
		'valence', 'tempo', 'explicit', 'complexity']
	to_plot_y = ['popularity_abs', 'popularity_rel']
	# plots these against each other to see if they are colinear
	to_plot_colin = ['acousticness', 'loudness', 'complexity']
	# list of models for which detailed stats are wanted
	to_plot_stats = ['popularity_abs ~ complexity']

	to_plot = (to_plot_x, to_plot_y, to_plot_colin, to_plot_stats)
	# creates summary files for all well-performing and queried models
	full_sum = True

	controller = Controller(getData, doPreprocess, auto, log, custom, full_sum, to_plot)
	controller.performActions()
	

if __name__ == "__main__":
	main()