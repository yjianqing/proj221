import pandas as pd
import numpy as np
import collections
from constants import dateToIdxMapping, typeOfSaleToWeightMapping
from datetime import datetime
import re

NUMBER_OF_COMPARISONS = 5

### Load File ###
filename = '../data/privateTrain.csv'
data = pd.read_csv(filename, index_col=0)

### Add Month Int ###
data["monthYearInt"] = np.nan
for index, row in data.iterrows():
	month = row['month']
	data.set_value(index, 'monthYearInt', dateToIdxMapping[month])

### Add Month ###
data["monthInt"] = np.nan
for index, row in data.iterrows():
	month = row['month']
	monthArray = month.split('-')
	data.set_value(index, 'monthInt', monthArray[1])

### Add Price Per Sqm ###
data["pricePerSqm"] = np.nan
for index, row in data.iterrows():
	data.set_value(index, 'pricePerSqm', row['price'] * 1.0 / row['areaInSqm'])

### Add yearsOfTenure and monthsOfTenureLeft ###
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

INFINITY = 9999
pattern = re.compile(r"[0-9]+")
data["yearsOfTenure"] = np.nan
data["monthsOfTenureLeft"] = np.nan
for index, row in data.iterrows():
	tenure = row['tenure']
	saleMonth = row['month']
	if tenure == "Freehold" or tenure == "Freehols": # There was a typo in original csv
		tenureYears = INFINITY
		monthsLeft = INFINITY*12
	elif tenure == "Na" or tenure == "N.A.":
		tenureYears = INFINITY
		monthsLeft = INFINITY*12
	else:
		matches = re.findall(pattern, tenure)
		saleMatches = re.findall(pattern, saleMonth)
		totalMonths = int(matches[0]) * 12
		tenureYears = int(matches[0])
		if len(matches) == 4:
			monthsGone = diff_month(datetime(int(saleMatches[0]),int(saleMatches[1]),1), datetime(int(matches[3]),int(matches[2]),int(matches[1])))
		elif len(matches) == 2:
			monthsGone = diff_month(datetime(int(saleMatches[0]),int(saleMatches[1]),1), datetime(int(matches[1]),1,1))
		else:
			monthsGone = 0
		monthsLeft = max(0, totalMonths - monthsGone)
	data.set_value(index, 'yearsOfTenure', tenureYears)
	data.set_value(index, 'monthsOfTenureLeft', monthsLeft)

### Add Completion Year ###
UNKNOWN = -1
data["completionYear"] = np.nan
for index, row in data.iterrows():
	completionDate = row['completionDate']
	if completionDate == "Unknown" or completionDate == "Uncompleted" or completionDate == "Uncomplete":
		toAdd = UNKNOWN
	else:
		matches = re.findall(pattern, completionDate)
		toAdd = int(matches[-1])
	data.set_value(index, 'completionYear', toAdd)

### Add 5 Previous Transactions (PricePerSqm, Diff In Month, Diff In Floor, Diff In Land Area)
### From Same Project Name
projectMap = {}
for index, row in data.iterrows():
	projectName = row['projectName']
	monthYearInt = row['monthYearInt']
	floorNum = row['floorNum']
	areaInSqm = row['areaInSqm']
	pricePerSqm = row['pricePerSqm']
	if projectName == "N.A.":
		continue
	if projectName not in projectMap:
		projectMap[projectName] = {}
	if monthYearInt not in projectMap[projectName]:
		projectMap[projectName][monthYearInt] = []
	projectMap[projectName][monthYearInt].append({
		'floorNum': floorNum,
		'areaInSqm': areaInSqm,
		'pricePerSqm': pricePerSqm
	})

for project, value in projectMap.iteritems():
	od = collections.OrderedDict(sorted(value.items(), reverse=True))
	projectMap[project] = od

newColumnNames = ["prevTransPricePerSqm", "prevTransDiffInMonthYearInt", "prevTransDiffInFloorNum", "prevTransDiffInAreaInSqm"]

for i in range(NUMBER_OF_COMPARISONS):
	for name in newColumnNames:
		data[name + str(i)] = np.nan

for index, row in data.iterrows():
	projectName = row['projectName']
	monthYearInt = row['monthYearInt']
	floorNum = row['floorNum']
	areaInSqm = row['areaInSqm']
	if projectName == "N.A." or projectName not in projectMap:
		continue
	idx = 0
	for compMonthYearInt, transList in projectMap[projectName].iteritems():
		if compMonthYearInt >= monthYearInt:
			continue
		for trans in transList:
			data.set_value(index, newColumnNames[0] + str(idx), trans['pricePerSqm'])
			data.set_value(index, newColumnNames[1] + str(idx), monthYearInt - compMonthYearInt)
			data.set_value(index, newColumnNames[2] + str(idx), floorNum - trans['floorNum'])
			data.set_value(index, newColumnNames[3] + str(idx), areaInSqm - trans['areaInSqm'])
			idx += 1
			if idx == NUMBER_OF_COMPARISONS:
				break
		if idx == NUMBER_OF_COMPARISONS:
			break

	if index%1000 == 0:
		print index

newFilename = '../data/privateTrainAugmented.csv'
data.to_csv(newFilename)

