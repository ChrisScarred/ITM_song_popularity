# In This Moment Song Popularity Analysis, 2021 version

This project aims to identify the most important musical aspects of a song by ITM with regards to the song popularity.

This version is currently in development.

## Prequisities

- make and activate a python virtual environment
- do `python -m pip install -r requirements.txt`

## Usage

The module contains several sub-modules:

- `dataGetter.py` for obtaining the data
- `preprocess.py` to preprocess the data
- `analyses.py` to compose models and analyse their performance
- `utils.py` with certain generally useful functions
- `controller.py` to control the behaviour of submodules and link them together

To use the module, one only has to edit `main.py`, specifically these variables:

- boolean `getData`: False if data should not be obtained but instead used the last saved version
- boolean `doPreprocess`: False if data should not be preprocessed and instead the last saved version of preprocessed data should be used
- boolean `auto`: if true, automatically obtains missing data; adviced to keep True
- boolean `log`: it True, information on model analyses should be saved into file `results.log`. If False, all information is printed via the console.
- list `custom`: contains model strings (models) that should be analysed regardless whether they are considered good by the algorithm; can be empty
- boolean `fullSum`: if True, saves summaries of all models considered good and all models in the list `custom` to the folder `summaries` by the name of the model
- variables for plot generation:
  - list `to_plot_x`: a list of variables to plot against `to_plot_y` variables in a scatter plot
  - list `to_plot_y`: a list of variables to plot against `to_plot_x` variables in a scatter plot
  - list `to_plot_colin`: a list of variables to plot against each other in pairs to see if any of them appear to have a colinear relationship
  - list `to_plot_stats`: a list of model strings (models) for which statistic graphs should be generated and saved. These include absolute residuals graph, histogram of residuals, the probplot and residuals against order of collection.

Explanations/documentation of each sub-module is included in its respective `.py` file.
