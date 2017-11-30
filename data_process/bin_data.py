import numpy as np
import matplotlib.pyplot as plt

data_prefix = 'hdb'
prices_train = np.load(data_prefix+'_train_Y.npy')
prices_test = np.load(data_prefix+'_test_Y.npy')

###make bins based on training data, not test data
bin_width = 20000

smallest = int(np.min(prices_train))
largest = int(0.5+np.max(prices_train))
bins = range(smallest, largest+bin_width, bin_width)
classes_train = np.digitize(prices_train, bins)

n = len(prices_train)
required_samples = 1000
assert n > required_samples
max_bin = np.max(classes_train)

while np.count_nonzero(classes_train==max_bin) < required_samples:
	classes_train = [x if x != max_bin else x-1 for x in classes_train]
	max_bin = np.max(classes_train)

largest = int(0.5+np.max(prices_test))
bins = range(smallest, largest+bin_width, bin_width)
classes_test = [x if x <= max_bin else max_bin for x in np.digitize(prices_test, bins)]

np.save(data_prefix+'_binned_train_Y.npy', np.array(classes_train)-1) #make classes 0 indexed
np.save(data_prefix+'_binned_test_Y.npy', np.array(classes_test)-1)

# smallest = int(np.min(classes_train))
# largest = int(np.max(classes_train))
# plt.hist(classes_train, bins=range(smallest, largest+1))
# plt.show()