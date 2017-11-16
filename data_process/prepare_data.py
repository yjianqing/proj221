import pandas as pd
import numpy as np

def process_data(filename, base_savename, private, cols_to_use):
	data = pd.read_csv(filename, usecols=cols_to_use)

	#sort by "year-month"
	data = data.sort_values('month')

	data[['year', 'month']] = data['month'].apply(lambda x: pd.Series(x.split('-')))
	data['month'] = data['month'].apply(lambda x: int(x))
	data['year'] = data['year'].apply(lambda x: int(x))

	#not in place
	def convert_one_hots(data, names):
		for name in names:
			data = data.join(pd.get_dummies(data[name]))
			data = data.drop(name, axis=1)
		return data

	#in place
	def normalize_columns(data, names):
		for name in names:
			col = data[name]
			data[name] = (col - col.mean()) / (col.max() - col.min())
	
	if private:
		data = convert_one_hots(data, ['typeOfArea', 'propertyType', 'typeOfSale', 'typeOfHousing', 'area'])
		normalize_columns(data, ['latitude', 'longitude', 'year', 'yearsOfTenure', 'monthsOfTenureLeft', 'completionYear', 'priceIndex'])
		price_name = 'price'
	else:    
		data = convert_one_hots(data, ['town', 'flatType', 'flatModel'])
		normalize_columns(data, ['latitude', 'longitude', 'year', 'leaseStartDate', 'priceIndex', 'areaInSqm', 'floorNum', 'month'])
		price_name = 'resalePrice'
	
	prices = data[price_name].as_matrix()
	data = data.drop(price_name, axis=1)

	test_proportion = 0.01
	num_test = int(test_proportion*len(data.index))
	
	train_Y = prices[:-num_test]
	test_Y = prices[-num_test:]
	train_X = data.iloc[:-num_test]
	test_X = data.iloc[-num_test:]

	np.save(base_savename+'_train_X.npy', train_X)
	np.save(base_savename+'_train_Y.npy', train_Y)
	np.save(base_savename+'_test_X.npy', test_X)
	np.save(base_savename+'_test_Y.npy', test_Y)

process_data('hdbHousing.csv', 'hdb', False, cols_to_use = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 15])
# process_data('privateHousing.csv', 'private', True, cols_to_use = [4, 5, 6, 10, 13, 14, 19, 20, 21, 22, 23, 25, 26, 27, 28])
