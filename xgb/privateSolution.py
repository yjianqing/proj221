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
from constants import typeOfSaleToWeightMapping

### MODEL PARAMETERS ###
NUM_OF_XGB_MODELS = 11
START_FROM_SEED = 19
NUM_ESTIMATORS = 5000
CV_SIZE = 0.7
CATEGORICAL_COLUMNS = [3, 6, 9]
NUMERICAL_COLUMNS = [2, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]
NUMERICAL_COLUMNS += [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
TEST_CATEGORICAL_COLUMNS = [3, 5, 8]
TEST_NUMERICAL_COLUMNS = [2, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
TEST_NUMERICAL_COLUMNS += [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
TYPE_OF_SALE_COLUMN = [9]
TARGET_COLUMN = 23 # Price Per Sqm
CV_ERROR_EVAL = "rmse"
EXPERIMENT_NUMBER = 1
RESULT_FILENAME = "../results/xgbPrivate-" + str(EXPERIMENT_NUMBER) + ".csv"
NUMBER_OF_COMPARISONS = 5

testNames = ['projectName', 'address', 'areaInSqm', 'typeOfArea', 'contractDate', 
'propertyType', 'tenure', 'completionDate', 'typeOfSale', 'postalDistrict', 
'postalSector', 'postalCode', 'region', 'area', 'month', 'latitude', 'longitude', 
'floorNum', 'unitNum', 'priceIndex', 'monthYearInt', 'monthInt', 'yearsOfTenure', 
'monthsOfTenureLeft', 'completionYear']
newColumnNames = ["prevTransPricePerSqm", "prevTransDiffInMonthYearInt", 
"prevTransDiffInFloorNum", "prevTransDiffInAreaInSqm"]
for i in range(5):
	for name in newColumnNames:
		testNames.append(name + str(i))

### LOAD DATA ###
filename = '../data/privateTrainAugmented.csv'
data = pd.read_csv(filename, index_col=0)

### GET TEST DATA ###
testFilename = '../data/privateTestAugmented.csv'
testData = pd.read_csv(testFilename, index_col=0)

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

### GENERATE TESTING X ###
X1 = np.column_stack(columns)
X2 = arrayTest[:, TEST_NUMERICAL_COLUMNS]
XTest = np.concatenate((X1, X2), axis=1)

### GENERATE WEIGHTS FOR TRAIN ###
weights = array[:, TYPE_OF_SALE_COLUMN]

### SPLIT WEIGHTS INTO TRAIN AND TEST ###
for i in range(0, len(weights)):
	weights[i][0] = typeOfSaleToWeightMapping[weights[i][0]]
weights = np.array([item for sublist in weights for item in sublist])

### TRAINING ###
models = []
for seed in range(START_FROM_SEED):
	modelFilename = "../models/xgbPrivate-" + str(EXPERIMENT_NUMBER) + "-" + str(seed) + ".pickle.dat"
	with open(modelFilename, 'rb') as f:
		models.append(pickle.load(f))
for seed in range(START_FROM_SEED, START_FROM_SEED + NUM_OF_XGB_MODELS):
	print seed
	cv_size = CV_SIZE
	X_train, X_cv, y_train, y_cv, weights_train, weights_cv = train_test_split(X, Y, weights, test_size=cv_size, random_state=seed)
	eval_set = [(X_train, y_train), (X_cv, y_cv)]
	model = XGBRegressor(
		colsample_bytree=0.7,
		gamma=0.0,
		learning_rate=0.2,
		max_depth=8,
		# min_child_weight=1.5,
		n_estimators=NUM_ESTIMATORS,
		reg_lambda=1,
		subsample=0.7)
	model.fit(X_train, y_train, eval_metric=CV_ERROR_EVAL, eval_set=eval_set, verbose=True, sample_weight=weights_train)
	models.append(model)
	now = datetime.datetime.now()
	pickle.dump(model, open("../models/xgbPrivate-" + str(EXPERIMENT_NUMBER) + "-" + str(seed) + ".pickle.dat", "wb"))

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
testNames.append("Prediction")
if NUM_OF_XGB_MODELS > 1:
	testNames.append("Confidence")
df = pd.DataFrame(toSave, columns=testNames)
df.index += 9699

### MOVING FROM PRICE_PER_SQM TO PRICE
def getPredictedPrice(row):
	return row['Prediction'] * row['areaInSqm']
df['price'] = df.apply(getPredictedPrice, axis=1)
df = df[['price']]
df.index.name = 'index'
df.to_csv(RESULT_FILENAME)

