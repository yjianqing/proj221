import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pk

data_prefix = 'hdb_appended'
# data_prefix = 'condo'
# data_prefix = 'landed'
prices_train = np.load(data_prefix+'_train_Y.npy')
prices_test = np.load(data_prefix+'_test_Y.npy')

###make bins based on training data, not test data
bin_width = 20000
# bin_width = int(1e6)


if data_prefix is 'hdb_appended':
    smallest = int(np.min(prices_train))
    largest = int(0.5+np.max(prices_train))
else:
    smallest = int(np.percentile(prices_train,1))
    largest = int(np.percentile(prices_train,99))

bins = range(smallest, largest+bin_width, bin_width)
classes_train = np.digitize(prices_train, bins)

n = len(prices_train)
required_samples = 1000
assert n > required_samples
max_bin = np.max(classes_train)

while np.count_nonzero(classes_train==max_bin) < required_samples:
	classes_train = [x if x != max_bin else x-1 for x in classes_train]
	max_bin = np.max(classes_train)

if data_prefix is 'hdb_appended':
    largest = int(0.5+np.max(prices_test))
else:
    largest = int(np.percentile(prices_test,99))
    
bins = range(smallest, largest+bin_width, bin_width)
classes_test = [x if x <= max_bin else max_bin for x in np.digitize(prices_test, bins)]

bin_to_prices = defaultdict(list)
for i in range(len(classes_test)):
	c = classes_train[i]
	bin_to_prices[c].append(prices_train[i])

bin_to_average_price = {}
for bin in bin_to_prices:
	bin_to_average_price[bin] = np.mean(bin_to_prices[bin])

np.save(data_prefix+'_binned_train_Y.npy', np.array(classes_train)-1) #make classes 0 indexed
np.save(data_prefix+'_binned_test_Y.npy', np.array(classes_test)-1)

f = open(data_prefix+'_bin_to_price.p', 'wb')
pk.dump(bin_to_average_price, f)
f.close()

# smallest = int(np.min(classes_train))
# largest = int(np.max(classes_train))
# plt.hist(classes_train, bins=range(smallest, largest+1))
# plt.show()
