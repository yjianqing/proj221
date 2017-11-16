import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

train_x = np.load('hdb_train_X.npy')
train_y = np.load('hdb_train_Y.npy')
test_x = np.load('hdb_test_X.npy')
test_y = np.load('hdb_test_Y.npy')

batch_size = 512

def loss(x, y):
	return np.sqrt(np.mean(np.square(x - y)))

def mean_predictor(train_y, test_y):
	average = np.mean(train_y)
	i = 0
	n = test_y.shape[0]
	print "mean is {}".format(average)
	while i < n:
		batch = test_y[i:i+batch_size]
		predictions = np.full(len(batch), average)
		rms = loss(predictions, batch)
		print "Rms for current batch: {}".format(rms)
		i += batch_size

def least_squares_predictor(train_x, train_y, test_x, test_y):
	regr = linear_model.LinearRegression()
	regr.fit(train_x, train_y)
	predictions = regr.predict(test_x)
	rms = loss(predictions, test_y)
	print "Rms for linear regression: {}".format(rms)
	return predictions

def ridge_regression_predictor(train_x, train_y, test_x, test_y, alpha):
	regr = linear_model.Ridge(alpha=alpha)
	regr.fit(train_x, train_y)
	i = 0
	n = test_y.shape[0]
	all_predictions = []
	while i < n:
		batch_x = test_x[i:i+batch_size]
		batch_y = test_y[i:i+batch_size]
		predictions = regr.predict(batch_x)
		rms = loss(predictions, batch_y)
		print "Rms for current batch: {}".format(rms)
		all_predictions = np.append(all_predictions, predictions)
		i += batch_size
	return all_predictions

print "Linear regression:"
least_squares_predictions = least_squares_predictor(train_x, train_y, test_x, test_y)

percentage_error = np.divide(least_squares_predictions - test_y, test_y)
positives = [err for err in percentage_error if err >= 0]
negatives = [err for err in percentage_error if err < 0]

plt.hist([negatives, positives], color=['tab:blue', 'tab:orange'], bins=100)
plt.xlabel("Percentage error of predicted price")
plt.ylabel("Number of predictions per bin")
plt.title("Linear Least Squares prediction error on hdb dataset")
plt.show()
