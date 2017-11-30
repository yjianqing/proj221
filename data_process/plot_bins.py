import matplotlib.pyplot as plt
import numpy as np

bin_width = 20000
prices = np.load('hdb_train_Y.npy')

smallest = int(np.min(prices))
largest = int(0.5+np.max(prices))

plt.hist(prices, bins=range(smallest, largest+bin_width, bin_width))
plt.show()