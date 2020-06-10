import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy

itm_data = pd.read_csv('itm_songs_database_preprocessed.csv')

# just keeping this for later
to_plot = ['key', 'mode', 'time_signature', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness',
           'valence', 'tempo', 'explicit', 'complexity', 'popularity_abs']
for var in to_plot:
    plt.figure()
    sns.scatterplot(x = itm_data[var], y = itm_data['popularity_abs']).set(title = 'scatterplot of popularity against '+var)
    plt.show()

to_plot = []

formula_abs = "popularity_abs ~ complexity"

# backwards selection stuff
# for key in to_plot:
#     formula_abs = formula_abs + " + " + key
#
# model_abs = sm.formula.ols(formula=formula_abs, data=itm_data)
# model_abs_fitted = model_abs.fit()
#
# print(model_abs_fitted.summary())

# forward selection stuff
# to_plot = ['key', 'mode', 'time_signature', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness',
#            'speechiness',
#            'valence', 'tempo', 'explicit', 'complexity']
# for key in to_plot:
#     formula_str = "popularity_abs ~ " + key
#     model = sm.formula.ols(formula=formula_str, data=itm_data)
#     model_fitted = model.fit()
#     print(key, ": ", model_fitted.pvalues[1])

# as purely linear model, the following vars are statistically significant:
# acousticness, loudness, complexity
# very close are: danceability, (valence, explicit)

# let's try to make a working model out of it:

formula_str = "popularity_abs ~ complexity"
model_abs = sm.formula.ols(formula=formula_str, data=itm_data)
model_abs_fitted = model_abs.fit()
print(model_abs_fitted.summary())

# check for possible collinearity visually:
# to_check = ['acousticness', 'loudness', 'complexity']
# for x_var in to_check:
#     for y_var in to_check:
#         plt.figure()
#         sns.scatterplot(x = itm_data[x_var], y = itm_data[y_var]).set(title = x_var + " & " + y_var)
#         plt.show()

# visual inspection: possible collinearity between acoust. & loudness, loudness & complexity
# will adjust the model above accordingly

# okay, so these two actually completely fuck over the significance, the only interaction that is reasonably significant
# is between accousticness and complexity; R^2 gets slightly improved
