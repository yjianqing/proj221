### This is an XGB Ensemble Model

# These 2 lines are to import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import datetime
import statistics

import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from constants import enbloc, dateToIdxMapping, typeOfSaleToWeightMapping

### MAGIC NUMBERS ###
NUM_OF_XGB_MODELS = 2
CV_SIZE = 0.7
ROW_NUMBER = 901
NUMBER_OF_COMPARISONS = 20
USED_NUMBER_OF_COMPARISONS = 5
RESULT_FILENAME = "../results/xgbHdbWithComparison-row" + str(ROW_NUMBER) + ".csv"
CATEGORICAL_COLUMNS = [2]
NUMERICAL_COLUMNS = [6, 8, 10, 11, 12, 14, 17]
COLUMN_TO_ADD = 19
newColumnNames = ["prevTransPricePerSqm", "prevTransDiffInMonthYearInt", "prevTransDiffInFloorNum", "prevTransDiffInAreaInSqm", "prevTransDiffTenure", "prevTransDistance"]
for i in range(NUMBER_OF_COMPARISONS):	
	for name in newColumnNames:
		NUMERICAL_COLUMNS.append(COLUMN_TO_ADD)
		COLUMN_TO_ADD += 1
TARGET_COLUMN = 18
MONTH_COLUMN = [0]
CV_RANGE = ['2017-1', '2017-2', '2017-3']
TEST_RANGE = ['2017-4', '2017-5', '2017-6']
CV_ERROR_EVAL = "mae"
CV = False
TRAIN_TEST_RANGE = [
'2010-1', '2010-2', '2010-3', '2010-4', '2010-5', '2010-6',
'2010-7', '2010-8', '2010-9', '2010-10', '2010-11', '2010-12',
'2011-1', '2011-2', '2011-3', '2011-4', '2011-5', '2011-6',
'2011-7', '2011-8', '2011-9', '2011-10', '2011-11', '2011-12',
'2012-1', '2012-2', '2012-3', '2012-4', '2012-5', '2012-6',
'2012-7', '2012-8', '2012-9', '2012-10', '2012-11', '2012-12',
'2013-1', '2013-2', '2013-3', '2013-4', '2013-5', '2013-6',
'2013-7', '2013-8', '2013-9', '2013-10', '2013-11', '2013-12',
'2014-1', '2014-2', '2014-3', '2014-4', '2014-5', '2014-6',
'2014-7', '2014-8', '2014-9', '2014-10', '2014-11', '2014-12',
'2015-1', '2015-2', '2015-3', '2015-4', '2015-5', '2015-6',
'2015-7', '2015-8', '2015-9', '2015-10', '2015-11', '2015-12',
'2016-1', '2016-2', '2016-3', '2016-4', '2016-5', '2016-6',
'2016-7', '2016-8', '2016-9', '2016-10', '2016-11', '2016-12',
'2017-1', '2017-2', '2017-3', '2017-4', '2017-5', '2017-6']

### LOAD DATA ###
filename = '../../data/allHdbHousingForTraining4.csv'
names = ['month', 'town', 'flatType', 'block', 'streetName', 'storeyRange', 'areaInSqm', 'flatModel',
		'leaseStartDate', 'resalePrice', 'floorNum', 'latitude', 'longitude', 'postalCode', 'priceIndex',
		'lowerStoreyRange', 'upperStoreyRange', 'monthInt','pricePerSqm']
for i in range(NUMBER_OF_COMPARISONS):	
	for name in newColumnNames:
		names.append(name + str(i))
data = pd.read_csv(filename, skiprows=1, names=names)

# def convertMonth(row):
# 	month = row["month"]
# 	monthArray = month.split('-')
# 	return monthArray[0] + '-' + str(int(monthArray[1]))

# data['month'] = data.apply(convertMonth, axis=1)
data = data[data["month"].str.contains("$|^".join(TRAIN_TEST_RANGE))]

### SPLIT TO TRAINING AND TEST SETS ###
test = data[data["month"].str.contains("$|^".join(TEST_RANGE))]
print test

### SANITY CHECK ###
# peek = data.head(20)
# print peek

### PROCESS TRAINING DATA ###
array = data.values
arrayTest = test.values
X1 = array[:, CATEGORICAL_COLUMNS]
columns = []

# Convert categorical data to one-hot encoding
for i in range(0, X1.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X1[:,i])
	print label_encoder.classes_
	feature = feature.reshape(X1.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	columns.append(feature)

### GENERATE X ###
X1 = np.column_stack(columns)
X2 = array[:, NUMERICAL_COLUMNS]
X3 = array[:, MONTH_COLUMN]
X = np.concatenate((X1, X2, X3), axis=1)

### GENERATE Y ###
Y = array[:, [TARGET_COLUMN]] # Target: price
Y = Y.astype(float)
Y = np.concatenate((Y, X3), axis=1)

### SPLIT X AND Y INTO TRAIN AND TEST ###
if CV:
	XTest = X[np.where([any(x == item for x in TEST_RANGE) for item in X[:,-1]])]
	XTest = XTest[:, :-1]
	YTest = Y[np.where([any(x == item for x in TEST_RANGE) for item in Y[:,-1]])]
	YTest = YTest[:, 0]
	X = X[np.where([all(x != item for x in TEST_RANGE) for item in X[:,-1]])]
	Y = Y[np.where([all(x != item for x in TEST_RANGE) for item in Y[:,-1]])]
	XCV = X[np.where([any(x == item for x in CV_RANGE) for item in X[:,-1]])]
	XCV = XCV[:, :-1]
	YCV = Y[np.where([any(x == item for x in CV_RANGE) for item in Y[:,-1]])]
	YCV = YCV[:, 0]
	X = X[np.where([all(x != item for x in CV_RANGE) for item in X[:,-1]])]
	Y = Y[np.where([all(x != item for x in CV_RANGE) for item in Y[:,-1]])]
	X = X[:, :-1]
	Y = Y[:, 0]
else:
	XTest = X[np.where([any(x == item for x in TEST_RANGE) for item in X[:,-1]])]
	XTest = XTest[:, :-1]
	YTest = Y[np.where([any(x == item for x in TEST_RANGE) for item in Y[:,-1]])]
	YTest = YTest[:, 0]
	X = X[np.where([all(x != item for x in TEST_RANGE) for item in X[:,-1]])]
	X = X[:, :-1]
	Y = Y[np.where([all(x != item for x in TEST_RANGE) for item in Y[:,-1]])]
	Y = Y[:, 0]
	X_devtest, XTest, Y_devtest, YTest, _, arrayTest = train_test_split(XTest, YTest, arrayTest, test_size=0.9)

print len(array)
print len(X)
print len(XTest)
print len(X_devtest)

### TRAINING ###
models = []
for seed in range(0,NUM_OF_XGB_MODELS):
	print seed
	cv_size = CV_SIZE
	X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size=cv_size, random_state=seed)
	# eval_set = [(X_train, y_train), (X_cv, y_cv)]
	# model = XGBRegressor(
	# 	colsample_bytree=0.7,
	# 	gamma=0.0,
	# 	learning_rate=0.1,
	# 	max_depth=5,
	# 	# min_child_weight=1.5,
	# 	n_estimators=5000,
	# 	reg_lambda=1,
	# 	subsample=0.7)
	# model.fit(X_train, y_train, eval_metric=[CV_ERROR_EVAL], eval_set=eval_set, verbose=True)
	if CV:
		dtrain = xgb.DMatrix(X_train, y_train)
		dcv = xgb.DMatrix(X_cv, y_cv)
		ddevtest = xgb.DMatrix(XCV, YCV)
		dtest = xgb.DMatrix(XTest, YTest)
		eval_set = [(dtest, 'test'), (dcv, 'cv'), (ddevtest, 'devtest')]
	else:
		dtrain = xgb.DMatrix(X_train, y_train)
		dcv = xgb.DMatrix(X_cv, y_cv)
		ddevtest = xgb.DMatrix(X_devtest, Y_devtest)
		dtest = xgb.DMatrix(XTest, YTest)
		eval_set = [(dtest, 'test'), (dcv, 'cv'), (ddevtest, 'devtest')]
	params = {
		'objective': 'reg:linear',
		'booster': 'gbtree',
		'learning_rate': 0.1,
		'gamma': 0.0,
		'sumsample': 0.7,
		'reg_lambda': 1,
		'max_depth': 8,
		'colsample_bytree': 0.7,
		'eval_metric': ['rmse', 'mae']
		# 'colsample_bylevel':
		# 'alpha'
		#  
	}
	model = xgb.train(params, dtrain, num_boost_round=5, evals=eval_set, early_stopping_rounds=50)
	models.append(model)
	now = datetime.datetime.now()
	pickle.dump(model, open("../models/xgb-ensemble-hdb-" + str(ROW_NUMBER) + "-" + str(seed) + ".pickle.dat", "wb"))

###### COMMENT THIS SECTION OUT WHEN NO TEST_RANGE

### PROCESS TESTING DATA ###
allPredictions = []
for model in models:
	y_pred = model.predict(xgb.DMatrix(XTest))
	predictions = [round(value) for value in y_pred]
	allPredictions.append(predictions)
allPredictions = map(list, zip(*allPredictions))
meanPrediction = []
stdDeviation = []
for pred in allPredictions:
	meanPrediction.append(statistics.mean(pred))
	if NUM_OF_XGB_MODELS > 1:
		stdDeviation.append(statistics.stdev(pred))

# ### SAVE TESTING RESULT ###
print len(arrayTest)
print len(meanPrediction)
print len(stdDeviation)
if NUM_OF_XGB_MODELS > 1:
	toSave = np.column_stack((arrayTest, meanPrediction, stdDeviation))
else:
	toSave = np.column_stack((arrayTest, meanPrediction))
names.append("Prediction")
if NUM_OF_XGB_MODELS > 1:
	names.append("Confidence")
toSave = np.row_stack((names, toSave))
df = pd.DataFrame(toSave)
df.to_csv(RESULT_FILENAME)
