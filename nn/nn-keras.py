from __future__ import print_function

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import argparse

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_price(raw_predictions, bin_width = 20000):
	bin_prices = np.array(range(num_classes))*bin_width
	prices = np.dot(raw_predictions, bin_prices)
	return prices

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_file_prefix", type=str, default='../data_process/hdb')
args = parser.parse_args()
data_file_prefix = args.data_file_prefix

X_train = np.load(data_file_prefix+'_train_X.npy')
Y_train = np.load(data_file_prefix+'_binned_train_Y.npy')

X_test = np.load(data_file_prefix+'_test_X.npy')
Y_test = np.load(data_file_prefix+'_binned_test_Y.npy')

num_features = X_train.shape[1]
num_classes = np.max(Y_train) + 1
assert num_classes > np.max(Y_test)
# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

def run_instance(data, epochs=10, dropout=0.1, batch_size=32, num_hidden=1, hidden_layer_size=128, model_save_file='cached_NN.h5', train_verbose=0):
	X_train, Y_train, X_test, Y_test = data
	X_train, Y_train = unison_shuffle(X_train, Y_train)
	n = X_train.shape[0]//10
	X_train, X_valid = X_train[:-n], X_train[-n:]
	Y_train, Y_valid = Y_train[:-n], Y_train[-n:]

	def make_NN(hidden_layer_size, dropout, num_hidden):
		model = Sequential()
		model.add(Dense(hidden_layer_size, activation='relu', input_shape=(num_features,)))
		model.add(Dropout(dropout))
		for _ in range(num_hidden):
			model.add(Dense(hidden_layer_size, kernel_initializer='normal', activation='relu'))
			model.add(Dropout(dropout))
		model.add(Dense(num_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])	
		return model

	model = make_NN(hidden_layer_size, dropout, num_hidden)
	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=train_verbose,
	                    validation_data=(X_valid, Y_valid))
	score = model.evaluate(X_test, Y_test, verbose=0)

	print("With params: epochs={}, dropout={}, batch_size={}, num_hidden={}, hidden_layer_size={}"\
			.format(epochs, dropout, batch_size, num_hidden, hidden_layer_size))
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	raw_predictions = model.predict(X_test)
	predicted_prices = get_price(raw_predictions)
	test_prices = np.load(data_file_prefix+'_test_Y.npy')
	rms = np.sqrt(np.mean(np.square(predicted_prices-test_prices)))
	print("RMS on test data: {}".format(rms))
	mean_abs = np.mean(np.absolute(predicted_prices-test_prices))
	print("Mean absolute on test data: {}".format(mean_abs))

	model.save(model_save_file)

for num_hidden in range(4):
	for hidden_layer_size in [64, 128, 256, 512]:
		for batch_size in [32, 128, 512]:
			for epochs in [1, 10, 20, 50]:
				for dropout in [0., .1, .2, .3]:
					save_name = 'model-{},{},{},{},{}.h5'.format(epochs, dropout, batch_size, num_hidden, hidden_layer_size).replace('.', '')
					run_instance((X_train, Y_train, X_test, Y_test), epochs, dropout, \
									batch_size, num_hidden, hidden_layer_size, save_name)
