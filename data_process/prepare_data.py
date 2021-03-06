import pandas as pd
import numpy as np
import os


NEW_RANGE = ['2017-7', '2017-8', '2017-9']

def process_data(filename, base_savename, private, cols_to_use):
	saved_name = 'sorted_{}.csv'.format(base_savename)
	
	if os.path.isfile(saved_name):
		data = pd.read_csv(saved_name)
		print data.tail(10)
		print data.columns
	else:
		data = pd.read_csv(filename, usecols=cols_to_use)
		print len(data)
		data = data[data["month"].str.contains("|".join(NEW_RANGE)) == False]
		print len(data)
		if base_savename == 'condo':
			data = data[~data['propertyType'].str.contains("House")].reset_index(drop=True)
		elif base_savename == 'landed':
			data = data[data['propertyType'].str.contains("House")].reset_index(drop=True)
		#sort by "year-month"
		data = data.sort_values('month').reset_index(drop=True)

		data[['year', 'month']] = data['month'].apply(lambda x: pd.Series(x.split('-')))
		data['month'] = data['month'].apply(lambda x: int(x))
		data['year'] = data['year'].apply(lambda x: int(x))

		data.to_csv(saved_name, index=False)
	
	print len(data)
	data = data[data['year'] >= 2000]
	data = data.reset_index(drop=True)
	print len(data)

	#not in place
	def convert_one_hots(data, names):
		for name in names:
			print "Name: {}".format(name)
			print len(data.columns)
			data = data.join(pd.get_dummies(data[name]))
			data = data.drop(name, axis=1)
			print len(data.columns)
		return data

	#in place
	def normalize_columns(data, names):
		for name in names:
			col = data[name]
			data[name] = (col - col.mean()) / (col.max() - col.min())
	
	def find_test_start_idx(data, year, month):
		return data[(data.year == year) & (data.month == month)].index[0] 

	if private:
		test_start_idx = find_test_start_idx(data, 2017, 2)
		data['floorNum'] = data['floorNum'].fillna(0)
		data['completionYear'] = data['completionYear'].replace(-1, np.nan)
		data['completionYear'] = data['completionYear'].fillna(data['completionYear'].mean())
		data = convert_one_hots(data, ['typeOfArea', 'propertyType', 'typeOfSale', 'typeOfHousing', 'area'])
		normalize_columns(data, ['latitude', 'longitude', 'year', 'yearsOfTenure', 'monthsOfTenureLeft', 'completionYear', 'priceIndex', 'areaInSqm', 'floorNum', 'month'])
		price_name = 'price'
	
	else:
		test_start_idx = find_test_start_idx(data, 2017, 4)
		data = convert_one_hots(data, ['town', 'flatType', 'flatModel'])
		columns = ['latitude', 'longitude', 'year', 'leaseStartDate', 'priceIndex', 'areaInSqm', 'floorNum', 'month']
		if base_savename == 'hdb_appended':
			for i in range(appended_data_num):
				columns.append('prevTransPricePerSqm'+str(i))
				columns.append('prevTransDiffInMonthYearInt'+str(i))
				columns.append('prevTransDiffInFloorNum'+str(i))
				columns.append('prevTransDiffInAreaInSqm'+str(i))
				columns.append('prevTransDiffTenure'+str(i))
				columns.append('prevTransDistance'+str(i))
				data['prevTransPricePerSqm'+str(i)] = data['prevTransPricePerSqm'+str(i)].fillna(data['prevTransPricePerSqm'+str(i)].mean())
				data['prevTransDiffInMonthYearInt'+str(i)] = data['prevTransDiffInMonthYearInt'+str(i)].fillna(data['prevTransDiffInMonthYearInt'+str(i)].mean())
				data['prevTransDiffInFloorNum'+str(i)] = data['prevTransDiffInFloorNum'+str(i)].fillna(data['prevTransDiffInFloorNum'+str(i)].mean())
				data['prevTransDiffInAreaInSqm'+str(i)] = data['prevTransDiffInAreaInSqm'+str(i)].fillna(data['prevTransDiffInAreaInSqm'+str(i)].mean())
				data['prevTransDiffTenure'+str(i)] = data['prevTransDiffTenure'+str(i)].fillna(data['prevTransDiffTenure'+str(i)].mean())
				data['prevTransDistance'+str(i)] = data['prevTransDistance'+str(i)].fillna(data['prevTransDistance'+str(i)].mean())	
				

		normalize_columns(data, columns)
		print data.tail(10)
		price_name = 'resalePrice'

	prices = data[price_name].as_matrix()
	data = data.drop(price_name, axis=1)

	print "starting test index: {}".format(test_start_idx)
	train_X = data.iloc[:test_start_idx]
	train_Y = prices[:test_start_idx]
	test_X = data.iloc[test_start_idx:]
	test_Y = prices[test_start_idx:]

	np.save(base_savename+'_train_X.npy', train_X)
	np.save(base_savename+'_train_Y.npy', train_Y)
	np.save(base_savename+'_test_X.npy', test_X)
	np.save(base_savename+'_test_Y.npy', test_Y)

#process_data('hdbHousing.csv', 'hdb', False, cols_to_use = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 15])
# process_data('privateHousing.csv', 'condo', True, cols_to_use = [4, 5, 6, 10, 13, 14, 19, 20, 21, 22, 23, 25, 26, 27, 28])
# process_data('privateHousing.csv', 'landed', True, cols_to_use = [4, 5, 6, 10, 13, 14, 19, 20, 21, 22, 23, 25, 26, 27, 28])

cols = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 15]
appended_data_start = 20
appended_data_num = 5
appended_data_len = 6
cols += list(range(appended_data_start, appended_data_start + appended_data_num * appended_data_len))
process_data('allHdbHousingForTraining4.csv', 'hdb_appended', False, cols_to_use = cols)

