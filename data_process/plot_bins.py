from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


# bin_width = 20000
# data_prefix = 'hdb'

bin_width = int(1e6)
#data_prefix = 'condo'
data_prefix = 'landed'

prices = np.load(data_prefix + '_train_Y.npy')
if data_prefix is 'hdb':
    smallest = int(np.min(prices))
    largest = int(0.5+np.max(prices))
else:
    smallest = int(np.percentile(prices,1))
    largest = int(np.percentile(prices,99))

print(smallest)
print(largest)
plt.hist(prices, bins=range(smallest, largest+bin_width, bin_width))
plt.show()
