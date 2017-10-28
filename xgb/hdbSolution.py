import pandas as pd
import numpy as np
import pickle
import datetime
import statistics
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from constants import dateToIdxMapping

### MODEL PARAMETERS ###
NUM_OF_XGB_MODELS = 30
NUM_ESTIMATORS = 2500
CV_SIZE = 0.7
CATEGORICAL_COLUMNS = [1, 2]
NUMERICAL_COLUMNS = [6, 8, 10, 11, 12, 14, 17]
TEST_CATEGORICAL_COLUMNS = [1, 2]
TEST_NUMERICAL_COLUMNS = [6, 8, 9, 10, 11, 13, 16]
TARGET_COLUMN = 18
TEST_RANGE = ['2017-1', '2017-2', '2017-3', '2017-4', '2017-5', '2017-6']
CV_ERROR_EVAL = "rmse"
EXPERIMENT_NUMBER = 1
RESULT_FILENAME = "../results/xgbHdb-" + str(EXPERIMENT_NUMBER) + ".csv"

### TRAIN ONLY ON THIS RANGE ###
TRAIN_RANGE = [
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
'2016-7', '2016-8', '2016-9', '2016-10', '2016-11', '2016-12']

### LOAD TRAINING DATA ###
filename = '../data/hdbTrain.csv'
names = ['month', 'town', 'flatType', 'block', 'streetName', 'storeyRange', 'areaInSqm', 'flatModel',
		'leaseStartDate', 'resalePrice', 'floorNum', 'latitude', 'longitude', 'postalCode', 'priceIndex',
		'minFloorRange', 'maxFloorRange']
data = pd.read_csv(filename, skiprows=1, names=names)

data = data[data["month"].str.contains("$|^".join(TRAIN_RANGE))]

### ADD MONTH INT AS FEATURE ###
def addMonthInt(row):
	return dateToIdxMapping[row["month"]]

data['monthInt'] = data.apply(addMonthInt, axis=1)

### ADD PRICE PER SQM AS TARGET ###
def addPricePerSqm(row):
	return float(row['resalePrice']) / row['areaInSqm']

data['pricePerSqm'] = data.apply(addPricePerSqm, axis=1)

### GET TEST DATA ###
testFilename = '../data/hdbTest.csv'
testNames = ['month', 'town', 'flatType', 'block', 'streetName', 'storeyRange', 'areaInSqm', 'flatModel',
		'leaseStartDate', 'floorNum', 'latitude', 'longitude', 'postalCode', 'priceIndex',
		'minFloorRange', 'maxFloorRange']
testData = pd.read_csv(testFilename, skiprows=1, names=testNames)

### ADD MONTH INT TO TEST ###
testData['monthInt'] = testData.apply(addMonthInt, axis=1)

### PROCESS TRAINING DATA ###
array = data.values
X1 = array[:, CATEGORICAL_COLUMNS]
columns = []

### CONVERT CATEGORICAL DATA TO ONE-HOT ENCODING FOR TRAIN DATA ###
CATEGORICAL_COLUMN_NAMES = []
for i in range(0, X1.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X1[:,i])
	CATEGORICAL_COLUMN_NAMES.append(label_encoder.classes_)
	feature = feature.reshape(X1.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	columns.append(feature)

### GENERATE TRAINING X ###
X1 = np.column_stack(columns)
X2 = array[:, NUMERICAL_COLUMNS]
X = np.concatenate((X1, X2), axis=1)

### GENERATE TRAINING Y ###
Y = array[:, [TARGET_COLUMN]] # Target: price
Y = Y.astype(float)

### PROCESS TESTING DATA ###
arrayTest = testData.values
X1 = arrayTest[:, TEST_CATEGORICAL_COLUMNS]
columns = []

### CONVERT CATEGORICAL DATA TO ONE-HOT ENCODING FOR TEST DATA ###
idx = 0
for i in range(0, X1.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit(CATEGORICAL_COLUMN_NAMES[idx])
	feature = label_encoder.transform(X1[:,i])
	feature = feature.reshape(X1.shape[0], 1)
	onehot_encoder = OneHotEncoder(n_values=len(CATEGORICAL_COLUMN_NAMES[idx]), sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	columns.append(feature)
	idx += 1

### GENERATE TRAINING X ###
X1 = np.column_stack(columns)
X2 = arrayTest[:, TEST_NUMERICAL_COLUMNS]
XTest = np.concatenate((X1, X2), axis=1)

### TRAINING ###
models = []
for seed in range(0,NUM_OF_XGB_MODELS):
	print "Training Model Number %s" %seed
	cv_size = CV_SIZE
	X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size=cv_size, random_state=seed)
	eval_set = [(X_train, y_train), (X_cv, y_cv)]
	model = XGBRegressor(
		colsample_bytree=0.7,
		gamma=0.0,
		learning_rate=0.1,
		max_depth=5,
		n_estimators=NUM_ESTIMATORS,
		reg_lambda=1,
		subsample=0.7)
	model.fit(X_train, y_train, eval_metric=[CV_ERROR_EVAL], eval_set=eval_set, verbose=True)
	models.append(model)
	now = datetime.datetime.now()
	pickle.dump(model, open("../models/xgbHdb-" + str(EXPERIMENT_NUMBER) + "-" + str(seed) + ".pickle.dat", "wb"))

### PROCESS TESTING DATA ###
allPredictions = []
for model in models:
	y_pred = model.predict(XTest)
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
if NUM_OF_XGB_MODELS > 1:
	toSave = np.column_stack((arrayTest, meanPrediction, stdDeviation))
else:
	toSave = np.column_stack((arrayTest, meanPrediction))
testNames.append("Month Int")
testNames.append("Prediction")
if NUM_OF_XGB_MODELS > 1:
	testNames.append("Confidence")
df = pd.DataFrame(toSave, columns=testNames)

### MOVING FROM PRICE_PER_SQM TO PRICE
def getPredictedPrice(row):
	return row['Prediction'] * row['areaInSqm']
df['price'] = df.apply(getPredictedPrice, axis=1)
df = df[['price']]
df.index.name = 'index'
df.to_csv(RESULT_FILENAME)


