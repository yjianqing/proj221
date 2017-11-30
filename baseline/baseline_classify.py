import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def loss(x, y):
	return np.sqrt(np.mean(np.square(x - y)))

def mape(x, y):
    return np.median(np.abs((y-x)/y))

def mean_predictor(train_y, test_y):
	average = np.mean(train_y)
	predictions = np.full(len(test_y), average)
	rms = loss(predictions, test_y)
	print "Rms for mean prediction: {}".format(rms)
	mdape = mape(predictions, test_y)
	print "Mape for linear regression: {}".format(mdape)
	return predictions

def least_squares_predictor(train_x, train_y, test_x, test_y):
	regr = linear_model.LinearRegression()
	regr.fit(train_x, train_y)
	predictions = regr.predict(test_x)
	rms = loss(predictions, test_y)
	print "Rms for linear regression: {}".format(rms)
	mdape = mape(predictions, test_y)
	print "Mape for linear regression: {}".format(mdape)
	return predictions

def ridge_regression_predictor(train_x, train_y, test_x, test_y, alpha):
	regr = linear_model.Ridge(alpha=alpha)
	regr.fit(train_x, train_y)
	predictions = regr.predict(test_x)
	rms = loss(predictions, test_y)
	print "Rms for ridge regression: {}".format(rms)
	mdape = mape(predictions, test_y)
	print "Mape for linear regression: {}".format(mdape)
	return predictions

def plot_least_squares_predictions(basename):
	train_x = np.load('{}_train_X.npy'.format(basename))
	train_y = np.load('{}_train_Y.npy'.format(basename))
	test_x = np.load('{}_test_X.npy'.format(basename))
	test_y = np.load('{}_test_Y.npy'.format(basename))

	print "Linear regression:"
	least_squares_predictions = least_squares_predictor(train_x, train_y, test_x, test_y)

	percentage_error = 100*np.divide(least_squares_predictions - test_y, test_y)
	positives = [err for err in percentage_error if err >= 0]
	negatives = [err for err in percentage_error if err < 0]
	if max(positives) > 100 or min(negatives) < -100:
		plt.xlim(xmin=-100, xmax=100)
	binwidth = 1
	bins=range(int(min(negatives)), int(max(positives) + binwidth), binwidth)
	plt.hist([negatives, positives], color=['tab:blue', 'tab:orange'], bins=bins)
	plt.xlabel("Percentage of overprediction")
	plt.ylabel("Number of records")
	plt.title("Linear Least Squares prediction error on {} dataset".format(basename))
	plt.savefig("{}_linear_regr_pred_error.png".format(basename))
	plt.show()

plot_least_squares_predictions('hdb')
plot_least_squares_predictions('condo')
plot_least_squares_predictions('landed')
