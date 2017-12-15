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
NUM_OF_UNITS = 1
MAX_PRICE_PER_SQFT = 3000
MAX_PRICE = 5000000
DEFAULT_FLOOR = 1
CATEGORICAL_COLUMNS = [4, 9, 12]
# NUMERICAL_COLUMNS = [3, 20, 21, 22, 23, 24, 25, 26, 28]
NUMERICAL_COLUMNS = [3, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 28, 29]
NUMERICAL_COLUMNS += [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
MONTH_COLUMN = [19]
TYPE_OF_SALE_COLUMN = [12]
TARGET_COLUMN = 6 # Price Per Sqm
CV_ERROR_EVAL = ["mae", "rmse"]
NUM_OF_XGB_MODELS = 30
ROW_NUMBER = 902
CV_SIZE = 0.7
RESULT_FILENAME = "../results/xgbEnsembleWithComparison-row" + str(ROW_NUMBER) + ".csv"
CV_RANGE = ["2017-4", "2017-5", "2017-6"]
TEST_RANGE = ["2017-7", "2017-8", "2017-9"]
PROPERTY_TYPES = ['Condominium', 'Apartment', 'Executive Condominium']
CV = False
# PROPERTY_TYPES = ['Semi-Detached House', 'Terrace House', 'Detached House']
# TEST_RANGE = []
NUMBER_OF_COMPARISONS = 5
# MIN_SOLD_PER_PROJECT = 10

newColumnNames = ["prevTransPricePerSqm", "prevTransDiffInMonthYearInt", "prevTransDiffInFloorNum", "prevTransDiffInAreaInSqm"]

### LOAD DATA ###
filename = '../../data/allPrivateHousingWithComparison11.csv'
names = ['projectName', 'address', 'numOfUnits', 'areaInSqm', 'typeOfArea', 'price', 
'pricePerSqm', 'pricePerSqft', 'contractDate', 'propertyType', 'tenure',
'completionDate', 'typeOfSale', 'typeOfHousing', 'postalDistrict', 'postalSector', 
'postalCode', 'region', 'area', 'month', 'latitude', 'longitude', 'floorNum', 'unitNum',
'yearsOfTenure', 'monthsOfTenureLeft', 'priceIndex', 'completionYear', 'monthYearInt', 'monthInt']
for i in range(5):
	for name in newColumnNames:
		names.append(name + str(i))

data = pd.read_csv(filename, skiprows=1, names=names)
data = data[data["numOfUnits"] == NUM_OF_UNITS]
data = data[data["projectName"].str.contains("|".join(enbloc)) == False]
data = data[data["propertyType"].str.contains("|".join(PROPERTY_TYPES))]
print data.propertyType.unique()
test = data[data["month"].str.contains("$|^".join(TEST_RANGE))]
print test

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
print X.shape

### GENERATE WEIGHTS ###
weights = array[:, TYPE_OF_SALE_COLUMN]
weights = np.concatenate((weights, X3), axis=1)
print weights.shape

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
print X.shape

### SPLIT WEIGHTS INTO TRAIN AND TEST ###
for i in range(0, len(weights)):
	weights[i][0] = typeOfSaleToWeightMapping[weights[i][0]]
weights_traincv = weights[np.where([all(x != item for x in TEST_RANGE) for item in weights[:,-1]])]
weights_test = weights[np.where([any(x == item for x in TEST_RANGE) for item in weights[:,-1]])]
if CV:
	weights_CV = weights_traincv[np.where([all(x != item for x in CV_RANGE) for item in weights[:,-1]])]
else:
	X_devtest, XTest, Y_devtest, YTest, _, arrayTest, weights_devtest, weights_test = train_test_split(XTest, YTest, arrayTest, weights_test, test_size=0.9)
weights_traincv = weights_traincv[:, 0]
weights_devtest = weights_devtest[:, 0]
weights_test = weights_test[:, 0]

### TRAINING ###
models = []
for seed in range(0,NUM_OF_XGB_MODELS):
	print seed
	cv_size = CV_SIZE
	X_train, X_cv, y_train, y_cv, weights_train, weights_cv = train_test_split(X, Y, weights_traincv, test_size=cv_size, random_state=seed)
	# eval_set = [(X_train, y_train), (X_cv, y_cv)]
	if CV:
		# WEIGHTS NOT DONE PROPERTLY
		dtrain = xgb.DMatrix(X_train, y_train, weight=weights_train)
		dcv = xgb.DMatrix(X_cv, y_cv, weight=weights_cv)
		ddevtest = xgb.DMatrix(XCV, YCV)
		dtest = xgb.DMatrix(XTest, YTest, weight=weights_test)
		eval_set = [(dtest, 'test'), (dcv, 'cv'), (ddevtest, 'devtest')]
	else:
		dtrain = xgb.DMatrix(X_train, y_train, weight=weights_train)
		dcv = xgb.DMatrix(X_cv, y_cv, weight=weights_cv)
		ddevtest = xgb.DMatrix(X_devtest, Y_devtest, weight=weights_devtest)
		dtest = xgb.DMatrix(XTest, YTest, weight=weights_test)
		eval_set = [(dtest, 'test'), (dcv, 'cv'), (ddevtest, 'devtest')]

	# model = XGBRegressor(
	# 	colsample_bytree=0.7,
	# 	gamma=0.0,
	# 	learning_rate=0.2,
	# 	max_depth=8,
	# 	# min_child_weight=1.5,
	# 	n_estimators=5000,
	# 	reg_lambda=1,
	# 	subsample=0.7)
	# model.fit(X_train, y_train, eval_metric=CV_ERROR_EVAL, eval_set=eval_set, verbose=True, sample_weight=weights_train)
	params = {
		'objective': 'reg:linear',
		'booster': 'gbtree',
		'learning_rate': 0.1,
		'gamma': 0.0,
		'sumsample': 0.7,
		'reg_lambda': 1,
		'max_depth': 8,
		'colsample_bytree': 0.7,
		'eval_metric': ['rmse', 'mae'],
		'colsample_bylevel': 0.7,
		# 'alpha'
		#  
	}
	model = xgb.train(params, dtrain, num_boost_round=5000, evals=eval_set) # , early_stopping_rounds=03
	models.append(model)
	now = datetime.datetime.now()
	pickle.dump(model, open("../models/xgb-ensemble-with-comparison-" + str(ROW_NUMBER) + "-" + str(seed) + ".pickle.dat", "wb"))


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

### SAVE TESTING RESULT ###
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

