#!/usr/bin/env python
# coding: utf-8

print('initializing')
# RF classifier model
from sklearn.ensemble import RandomForestClassifier
# function to split training data to traing and test sets
from sklearn.model_selection import train_test_split
# metrics for RF model results
from sklearn import metrics
# accuracy score
from sklearn.metrics import accuracy_score
# classification report
from sklearn.metrics import classification_report

# import numpy
import numpy, sys
# import gdal and other raster related packages
from osgeo import gdal, osr, ogr
# import pandas
import pandas
# import rasterio
import rasterio

# hyperparameter tuning
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# extracting and plotting a single decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# export figure of a single tree
from sklearn.tree import export_graphviz
# convert figure to png
from subprocess import call

from sklearn.metrics import confusion_matrix

# visualize important features
import seaborn

# save RF model for future use
import pickle

# load first raster file of test data for training and testing RF model for Pamir Alay range
# test_data_raster = 'smb://dartfs-hpc/rc/lab/H/Hawley/r/vhalvorson/model_inputs/test1a/land7_training_part1_final.tif'
test_data_raster = 'model_inputs/test05b2/land7_training_part1_final.tif'
# second raster file of test data
test_data_2 = 'model_inputs/test05b2/land7_training_part2_final.tif'
# load test DEM into new variable
test_dem = 'model_inputs/test05b2/dem_clip.tif'
# load test velocity into new variable
velocity_test = 'model_inputs/test05b2/velocity_clipped.tif'
# load raster file of test classification band
test_class_band = 'model_inputs/test05b2/inventory_RGI6_clip.tif'
# load test slope into new variable
test_load_slope = 'model_inputs/test05b2/slope_clip.tif'
# load test aspect into new variable
test_load_aspect = 'model_inputs/test05b2/aspect_clip.tif'

# define classification report csv
class_report = 'class_report_05b2.csv'
# define feature importance csv
feature_importance = 'feature_import_05b2.csv'
# define confusion matrix csv
conf_matrix_csv = 'confusion_matrix_05b2.csv'
# classification features = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'NDVI', 'NDWI', 'NDSI', 'elevation', 'velocity']
features = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_7', 'NDSI', 'elevation', 'slope', 'aspect', 'velocity']


#smb://dartfs-hpc/rc/lab/H/HawleyR/vhalvorson/model_outputs


print('loading training data part 1')
# # load raster file of test data for training and testing RF model for Pamir Alay range
# test_data_raster = 'PamirAlayMosaic_smallerFloat.tif'
# open test raster
test_data_raster = gdal.Open(test_data_raster, gdal.GA_ReadOnly)
# get test raster info
rows = test_data_raster.RasterYSize
cols = test_data_raster.RasterXSize
bands = test_data_raster.RasterCount
print('rows:', rows, 'columns:', cols, 'bands:', bands)
# read test data as an array (band, row, column)
test_array = test_data_raster.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_array = numpy.stack(test_array, axis = 2)
# multiply row and column elements to make 2D array (row*column, band)
test_array = numpy.reshape(test_array, [rows*cols, bands])
# define dataframe (table) for test array with columns equal to columns
test_data = pandas.DataFrame(test_array, columns=["band_1","band_2","band_3"])

print('loading training data part 2')
# # load raster file of test data for training and testing RF model for Pamir Alay range
# test_data_raster = 'PamirAlayMosaic_smallerFloat.tif'
# open test raster
test_data_2 = gdal.Open(test_data_2, gdal.GA_ReadOnly)
# get test raster info
rows = test_data_2.RasterYSize
cols = test_data_2.RasterXSize
bands = test_data_2.RasterCount
print('rows:', rows, 'columns:', cols, 'bands:', bands)
# read test data as an array (band, row, column)
test_array_2 = test_data_2.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_array_2 = numpy.stack(test_array_2, axis = 2)
# multiply row and column elements to make 2D array (row*column, band)
test_array_2 = numpy.reshape(test_array_2, [rows*cols, bands])
# define dataframe (table) for test array with columns equal to columns
test_data_2 = pandas.DataFrame(test_array_2, columns=["band_4","band_5","band_7"])
# append part 2 of test data to test_data
test_data["band_4"] = test_data_2["band_4"]
test_data["band_5"] = test_data_2["band_5"]
test_data["band_7"] = test_data_2["band_7"]

print('calculating NDSI')
# NDSI for test_data and amend to dataframe
# NDSI = (Green - SWIR)/(Green + SWIR) = (band_2 - band_5)/(band_2 + band_5)

# select the green and SWIR colums in the test_data table
green = test_data['band_2']
swir = test_data_2['band_5']

test_data["NDSI"] = (green - swir)/(green + swir)


print('loading test elevation')
# add elevation, slope and aspect to test data
# # load DEM into new variable
# test_dem = 'PamirAlayDEM_smaller.tif'
# open DEM with GDAL
test_dem = gdal.Open(test_dem, gdal.GA_ReadOnly)
# get test DEM info
rows = test_dem.RasterYSize
cols = test_dem.RasterXSize
bands = test_dem.RasterCount
print('rows:', rows, 'columns:', cols, 'bands:', bands)
# read test DEM as an array (band, row, column)
test_elevation = test_dem.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_elevation = numpy.stack(test_elevation)
# multiply row and column elements to make 2D array (row*column, band)
test_elevation = numpy.reshape(test_elevation, [rows*cols, bands])
# define dataframe (table) for test elevation data
test_elev = pandas.DataFrame(test_elevation, columns=["elevation"])


print('loading test velocity')
# # load velocity into new variable
# velocity_test = 'velocity_PamirAlay.tif'
# open velocity with GDAL
test_vel = gdal.Open(velocity_test, gdal.GA_ReadOnly)
# get test DEM info
vel_rows = test_vel.RasterYSize
vel_cols = test_vel.RasterXSize
vel_bands = test_vel.RasterCount
print('rows:', rows, 'columns:', cols, 'bands:', bands)
# read test velocity as an array (band, row, column)
test_velocity = test_vel.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_velocity = numpy.stack(test_velocity)
# multiply row and column elements to make 2D array (row*column, band)
test_velocity = numpy.reshape(test_velocity, [vel_rows*vel_cols, vel_bands])
# define dataframe (table) for test velocity data
test_v = pandas.DataFrame(test_velocity, columns=["velocity"])

print('loading test slope')
# open slope with GDAL
test_load_slope = gdal.Open(test_load_slope, gdal.GA_ReadOnly)
# get test slope info
slope_rows = test_load_slope.RasterYSize
slope_cols = test_load_slope.RasterXSize
slope_bands = test_load_slope.RasterCount
# read slope ad an array (band, row, col)
test_array_slope = test_load_slope.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_array_slope = numpy.stack(test_array_slope)
# multiply row and column elements to make 2D array (row*column, band)
test_array_slope = numpy.reshape(test_array_slope, [slope_rows*slope_cols, slope_bands])
# define dataframe (table) for test slope data
test_slope = pandas.DataFrame(test_array_slope, columns=["slope"])


print('loading test aspect')
# open aspect with GDAL
test_load_aspect = gdal.Open(test_load_aspect, gdal.GA_ReadOnly)
# get test aspect info
aspect_rows = test_load_aspect.RasterYSize
aspect_cols = test_load_aspect.RasterXSize
aspect_bands = test_load_aspect.RasterCount
# read aspect as an array (band, row, column)
test_array_aspect = test_load_aspect.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
test_array_aspect = numpy.stack(test_array_aspect)
# multiply row and column elements to make 2D array (row*column, band)
test_array_aspect = numpy.reshape(test_array_aspect, [aspect_rows*aspect_cols, aspect_bands])
# define dataframe (table) for test aspect data
test_aspect = pandas.DataFrame(test_array_aspect, columns=["aspect"])


print('loading test classification data')
# add classififcation info to test data
# # load raster file of test classification band
# test_class_band = 'PamirAlay_second_glacinv.tif'
# open test class band and read as array
test_class_band = gdal.Open(test_class_band, gdal.GA_ReadOnly)
rows = test_class_band.RasterYSize
cols = test_class_band.RasterXSize
bands = test_class_band.RasterCount
# read test class band as an array (band, row, column)
class_band = test_class_band.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
class_band = numpy.stack(class_band)
# multiply row and column elements to make 2D array (row*column, band)
class_band = numpy.reshape(class_band, [rows*cols, bands])
# define dataframe (table) for test classification band
test_class = pandas.DataFrame(class_band, columns=["class"])


# append elevation to test_data
test_data["elevation"] = test_elev
# append slope to test_data
test_data["slope"] = test_slope
# append aspect to test_data
test_data["aspect"] = test_aspect
# append velocity to test_data
test_data["velocity"] = test_v
# append class to test_data
test_data["class"] = test_class

# print test_data
print(test_data)

print(test_v)


# 70/30 split training data into to training and testing sets
# X are features (inputs), y are labels (outputs)
# with the test data csv, include only landsat band data for X, include only classificaion labels for y
X = test_data.drop(['class'], axis = 'columns') # features
y = test_data['class'] # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# # tune hyperparameters - number of decision trees (n_estimators), depth of each tree (max_depth)
# # define distributions (range) of parameters to be tested
# param_dist = {'n_estimators': randint(20,100),
#               'max_depth': randint(1,20)}

# # create variable for a randon forest classifier
# model = RandomForestClassifier()

# # use random search to find the best hyperparameters by sampling 10 parameter settings
# rand_search = RandomizedSearchCV(model,
#                                  param_distributions = param_dist,
#                                  n_iter = 10,
#                                  n_jobs = 8,
#                                  verbose = 4)

# # fit the random search to the test data
# rand_search.fit(X_train, y_train)

# # create a variable for the best model parameters
# rf_model = rand_search.best_estimator_

# # print the best hyperparameters
# print('Best hyperparameters:', rand_search.best_params_)


## this part is only necessary if you don't also do hyperparameter tuning 
## otherwise you could just use the trained model from hyperparameter tuning (rf_model in the code above)
# best hyperparameters: max_depth: 19, n_estimators: 85
rf_model = RandomForestClassifier(n_estimators = 55, max_depth = 18)
# rf_model = RandomForestClassifier(n_estimators = 5, max_depth = 2)

print('training model')
# train the model (w/best parameters) to fit X and y training data
# determine model accuracy score w/ X and y test data
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
print('done training')


print('creating classification report data')
# define y_pred variable of predicted labels classified from X test data (needed for classification report)
y_pred = rf_model.predict(X_test)

# define class name and feature variables for classification results
class_names = ['ice', 'ice-free', 'no-data']
# features = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'NDVI', 'NDWI', 'NDSI', 'elevation', 'velocity']
features = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_7', 'NDSI', 'elevation', 'slope', 'aspect', 'velocity']

# print classification report of model test results
print(classification_report(y_test, y_pred, target_names = class_names))

# export classification report as csv file
# create dictionary of class report
report_dict = classification_report(y_test, y_pred, target_names = class_names, output_dict=True)
# constructe dataframe and transpose it (switch cols and rows)
report_df =  pandas.DataFrame(report_dict).transpose()
# generate csv file with report dataframe
report_df.to_csv(class_report)


print('creating feature importance data')
# determine importance of each feature in classifying test data 
feature_import = pandas.Series(rf_model.feature_importances_, index = features).sort_values(ascending = False)
# feature_import = pandas.Series(rf_model.feature_importances_, index = features).sort_values(ascending = False)
#feature_import = pandas.Series(best_rf.feature_importances_).sort_values(ascending = False)

print('exporting confusion matrix')
# generate confusion matrix using predicted labels and actual values
conf_matrix = confusion_matrix(y_test, y_pred)
# convert matrix to dataframe
df_conf_matrix = pandas.DataFrame(conf_matrix)
# export confusion matrix to csv file
df_conf_matrix.to_csv(conf_matrix_csv)

# export feature importance as csv file
feature_import.to_csv(feature_importance)

# save the model using pickel
pickle.dump(rf_model, open('rf_model_05b2.pkl', 'wb'))