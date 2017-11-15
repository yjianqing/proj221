import pandas as pd
import numpy as np

def process_data(filename, cols_to_use, is_train=True):
	data = pd.read_csv(filename, usecols=cols_to_use)
	data[['year', 'month']] = data['month'].apply(lambda x: pd.Series(x.split('-')))
	data['month'] = data['month'].apply(lambda x: int(x))
	data['year'] = data['year'].apply(lambda x: int(x))

	#not in place
	def convert_one_hots(data, names):
		for name in names:
			data = data.join(pd.get_dummies(data[name]))
			data = data.drop(name, axis=1)
		return data

	data = convert_one_hots(data, ['town', 'flatType', 'flatModel'])

	#in place
	def normalize_columns(data, names):
		for name in names:
			col = data[name]
			data[name] = (col - col.mean()) / (col.max() - col.min())
	normalize_columns(data, ['latitude', 'longitude', 'year', 'leaseStartDate', 'priceIndex'])
	if is_train:
		prices = data['price'].as_matrix()
		data = data.drop('price', axis=1)
		return data.as_matrix(), prices
	else:
		return data.as_matrix()

def process_prices(filename):
	cols_to_use = [1]
	prices = pd.read_csv(filename, usecols=cols_to_use)
	return prices.as_matrix()

#training_X, training_Y = process_data('hdbTrain.csv', cols_to_use = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 15])
#np.save('hdbTrain_X.npy', training_X)
#np.save('hdbTrain_Y.npy', training_Y)

test_X = process_data('hdbTest.csv', cols_to_use = [1, 2, 3, 7, 8, 9, 10, 11, 12, 14], is_train=False)
np.save('hdbTest_X.npy', test_X)

test_Y = process_prices('hdbTestY.csv')
np.save('hbdTest_Y.npy', test_Y)
