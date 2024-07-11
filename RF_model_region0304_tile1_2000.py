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

# import trained model from pickle
import pickle

# load first Landsat file of input data for glacier classification
targetRaster = 'model_inputs/region0304/tile1_2000/land7_part1_masked.tif'
# second Landsat file of input data
targetRaster2 = 'model_inputs/region0304/tile1_2000/land7_part2_masked.tif'
# load DEM into new variable
target_dem = 'model_inputs/region0304/tile1_2000/dem_masked.tif'
# load velocity into new variable
velocity_target = 'model_inputs/region0304/tile1_2000/velocity_masked.tif'
# load slope into new variable
target_load_slope = 'model_inputs/region0304/tile1_2000/slope_masked.tif'
# load aspect into new variable
target_load_aspect = 'model_inputs/region0304/tile1_2000/aspect_masked.tif'

# define output raster location for target classification results
output_raster = 'model_outputs/region0304_tile1_2000_masked.tif'

print('loading input data')
# open first input raster
targetRaster = gdal.Open(targetRaster, gdal.GA_ReadOnly)
# get target raster info
rows = targetRaster.RasterYSize
cols = targetRaster.RasterXSize
bands = targetRaster.RasterCount
geo_transform = targetRaster.GetGeoTransform()
projection = targetRaster.GetProjectionRef()
# read target data as an array (band, row, column)
target_array = targetRaster.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_array = numpy.stack(target_array, axis = 2)
# multiply row and column elements to make 2D array (row*column, band)
target_array = numpy.reshape(target_array, [rows*cols, bands])
# define dataframe (table) for target array with columns equal to landsat bands
target_data = pandas.DataFrame(target_array, columns=["band_1","band_2","band_3"])

print('loading input data part 2')
# open second input raster
targetRaster2 = gdal.Open(targetRaster2, gdal.GA_ReadOnly)
# get target raster info
rows = targetRaster2.RasterYSize
cols = targetRaster2.RasterXSize
bands = targetRaster2.RasterCount
geo_transform = targetRaster2.GetGeoTransform()
projection = targetRaster2.GetProjectionRef()
# read target data as an array (band, row, column)
target_array2 = targetRaster2.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_array2 = numpy.stack(target_array2, axis = 2)
# multiply row and column elements to make 2D array (row*column, band)
target_array2 = numpy.reshape(target_array2, [rows*cols, bands])
# define dataframe (table) for target array with columns equal to landsat bands
target_data2 = pandas.DataFrame(target_array2, columns=["band_4","band_5","band_7"])
# append part 2 of input data to first part
target_data["band_4"] = target_data2["band_4"]
target_data["band_5"] = target_data2["band_5"]
target_data["band_7"] = target_data2["band_7"]

print('calculating NDSI')
# select the green and SWIR colums in the target_data table
green = target_data['band_2']
swir = target_data['band_5']

# calculate NDSI for target_data 
target_data["NDSI"] = (green - swir)/(green + swir)

print('loading elevation')
# open DEM with GDAL
target_dem = gdal.Open(target_dem, gdal.GA_ReadOnly)
# get test DEM info
rows = target_dem.RasterYSize
cols = target_dem.RasterXSize
bands = target_dem.RasterCount
print('rows:', rows, 'columns:', cols, 'bands:', bands)
# read test DEM as an array (band, row, column)
target_elevation = target_dem.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_elevation = numpy.stack(target_elevation)
# multiply row and column elements to make 2D array (row*column, band)
target_elevation = numpy.reshape(target_elevation, [rows*cols, bands])
# define dataframe (table) for test elevation data
target_elev = pandas.DataFrame(target_elevation, columns=["elevation"])

print('loading velocity')
# open velocity with GDAL
velocity_target = gdal.Open(velocity_target, gdal.GA_ReadOnly)
# get test velocity info
vel_rows = velocity_target.RasterYSize
vel_cols = velocity_target.RasterXSize
vel_bands = velocity_target.RasterCount
print('rows:', vel_rows, 'columns:', vel_cols, 'bands:', vel_bands)
# read test velocity as an array (band, row, column)
target_velocity = velocity_target.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_velocity = numpy.stack(target_velocity)
# multiply row and column elements to make 2D array (row*column, band)
target_velocity = numpy.reshape(target_velocity, [vel_rows*vel_cols, vel_bands])
# define dataframe (table) for test velocity data
target_v = pandas.DataFrame(target_velocity, columns=["velocity"])

print('loading target slope')
# open slope with GDAL
target_load_slope = gdal.Open(target_load_slope, gdal.GA_ReadOnly)
# get test slope info
slope_rows = target_load_slope.RasterYSize
slope_cols = target_load_slope.RasterXSize
slope_bands = target_load_slope.RasterCount
# read slope ad an array (band, row, col)
target_array_slope = target_load_slope.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_array_slope = numpy.stack(target_array_slope)
# multiply row and column elements to make 2D array (row*column, band)
target_array_slope = numpy.reshape(target_array_slope, [slope_rows*slope_cols, slope_bands])
# define dataframe (table) for test slope data
target_slope = pandas.DataFrame(target_array_slope, columns=["slope"])


print('loading target aspect')
# open aspect with GDAL
target_load_aspect = gdal.Open(target_load_aspect, gdal.GA_ReadOnly)
# get test aspect info
aspect_rows = target_load_aspect.RasterYSize
aspect_cols = target_load_aspect.RasterXSize
aspect_bands = target_load_aspect.RasterCount
# read aspect as an array (band, row, column)
target_array_aspect = target_load_aspect.ReadAsArray()
# restack array to change order of elements and move band to end (row, column, band)
target_array_aspect = numpy.stack(target_array_aspect)
# multiply row and column elements to make 2D array (row*column, band)
target_array_aspect = numpy.reshape(target_array_aspect, [aspect_rows*aspect_cols, aspect_bands])
# define dataframe (table) for test aspect data
target_aspect = pandas.DataFrame(target_array_aspect, columns=["aspect"])


# append elevation to target_data
target_data["elevation"] = target_elev
# append slope to target_data
target_data["slope"] = target_slope
# append aspect to target_data
target_data["aspect"] = target_aspect
# append velocity to target_data
target_data["velocity"] = target_v

# import trained random forest model from pickle
pickled_model = pickle.load(open('best_rf_model_05a2.pkl', 'rb'))

print('classifying target data')
# classify target data, variable stored as 1D array
target_prediction = pickled_model.predict(target_data)

# reshape target classification as 2D array with rows and columns as elements
target_class = target_prediction.reshape((rows, cols))


print('exporting target classification results')
# create Geotiff to export figure of classified target data
def createGeotiff(output_raster, target_class, geo_transform, projection):
    # create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = target_class.shape
    rasterDS = driver.Create(output_raster, cols, rows, 1, gdal.GDT_Int32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(target_class)
    rasterDS = None

# export classified image
createGeotiff(output_raster,target_class,geo_transform,projection)


