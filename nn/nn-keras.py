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
import matplotlib.pyplot as plt
import pickle as pk

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_file_prefix", type=str, default='../data_process/hdb')
args = parser.parse_args()
data_file_prefix = args.data_file_prefix

X_train = np.load(data_file_prefix+'_train_X.npy')
#X_train = np.delete(X_train, 7, axis=1)
Y_train = np.load(data_file_prefix+'_binned_train_Y.npy')

X_test = np.load(data_file_prefix+'_test_X.npy')
#X_test = np.delete(X_test, 7, axis=1)
Y_test = np.load(data_file_prefix+'_binned_test_Y.npy')

X_train_2 = np.load('../data_process/hdb'+'_train_X.npy')
Y_train_2 = np.load('../data_process/hdb'+'_binned_train_Y.npy')

X_test_2 = np.load('../data_process/hdb'+'_test_X.npy')
Y_test_2 = np.load('../data_process/hdb'+'_binned_test_Y.npy')
print(np.array_equal(X_train, X_train_2))

num_features = X_train.shape[1]
num_classes = np.max(Y_train) + 1

assert num_classes > np.max(Y_test)
# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

def run_instance(data, epochs=10, dropout=0.1, batch_size=32, num_hidden=1, hidden_layer_size=128, model_save_file='cached_NN.h5', train_verbose=1):
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
	model.save(model_save_file)
	return raw_predictions, history

def get_price(raw_predictions, test_prices, bin_width = 20000):
	print(bin_width)
	bin_prices = np.array(range(num_classes))*bin_width + 2*bin_width
	predicted_prices = np.dot(raw_predictions, bin_prices)
	rms = np.sqrt(np.mean(np.square(predicted_prices-test_prices)))
	print("RMS on test data: {}".format(rms))
	mean_abs = np.mean(np.absolute(predicted_prices-test_prices))
	print("Mean absolute on test data: {}".format(mean_abs))
	med_abs_per = np.median(np.absolute((predicted_prices-test_prices)/test_prices))
	print("Median absolute percentage on test data: {}".format(med_abs_per))

	###plot error for normal predictions
	percentage_error = 100*np.divide((predicted_prices - test_prices), test_prices)
	positives = [err for err in percentage_error if err >= 0]
	negatives = [err for err in percentage_error if err < 0]
	if max(positives) > 100 or min(negatives) < -100:
		plt.xlim(xmin=-100, xmax=100)
	
	bins=range(int(min(negatives)), int(max(positives) + 1))
	plt.hist([negatives, positives], color=['tab:blue', 'tab:orange'], bins=bins)
	plt.title("Neural net prediction error on HDB dataset")
	plt.xlabel("Percentage of overprediction")
	plt.ylabel("Number of records")
	plt.savefig("hdb_nn.png")
	plt.show()
	return predicted_prices

def conservative_predict(raw_predictions, test_prices, bin_width=20000):
	bin_prices = np.array(range(num_classes))*bin_width
	predicted_prices = np.dot(raw_predictions, bin_prices) + 2*bin_width
	num_predicted = 0
	num_skipped = 0
	rms = 0
	abs_err = 0
	confidences = np.max(raw_predictions, axis=1)
	thresh = np.median(confidences)
	percentage_error = []
	abs_percentage_error = []
	for i in range(len(predicted_prices)):
		if confidences[i] < thresh:
			num_skipped += 1
			predicted_prices[i] = -1
		else:
			num_predicted += 1
			diff = predicted_prices[i]-test_prices[i]
			percentage_error.append(100* diff / test_prices[i])
			abs_percentage_error.append(np.absolute(100* diff / test_prices[i]))
			rms += np.square(diff)
			abs_err += np.absolute(diff)

	rms /= num_predicted
	rms = np.sqrt(rms)
	abs_err /= num_predicted
	print("{} predictions made, {} skipped".format(num_predicted, num_skipped))
	print("RMS on test data: {}".format(rms))
	print("Mean absolute on test data: {}".format(abs_err))
	print("Median absolute percentage on test data: {}".format(np.median(abs_percentage_error)))

	###plot error for conservative predictions
	positives = [err for err in percentage_error if err >= 0]
	negatives = [err for err in percentage_error if err < 0]
	if max(positives) > 100 or min(negatives) < -100:
		plt.xlim(xmin=-100, xmax=100)
	binwidth = 1
	bins=range(int(min(negatives)), int(max(positives) + binwidth), binwidth)
	plt.hist([negatives, positives], color=['tab:blue', 'tab:orange'], bins=bins)
	plt.title("Neural net prediction error with filtering on HDB dataset")
	plt.xlabel("Percentage of overprediction")
	plt.ylabel("Number of records")
	plt.savefig("hdb_nn_filtered.png")
	plt.show()
	return predicted_prices

if 'hdb_appended' in data_file_prefix:
	print("hdb_appended")
	bin_width = 20000
	model_save_file = 'hdb_appended_model_saved.h5'
	epochs = 30
	dropout = 0.1
	batch_size = 256
	num_hidden = 2
	hidden_layer_size = 256
elif 'hdb' in data_file_prefix:
	print("hdb")
	bin_width = 20000
	model_save_file = 'hdb_model_saved.h5'
	epochs = 10
	dropout = 0.1
	batch_size = 256
	num_hidden = 2
	hidden_layer_size = 256
elif 'landed' in data_file_prefix:
	bin_width = int(1e6)
	model_save_file = 'landed_model_saved.h5'
	epochs = 10
	dropout = 0.1
	batch_size = 256
	num_hidden = 2
	hidden_layer_size = 256
elif 'condo' in data_file_prefix:
	model_save_file = 'condo_model_saved.h5'
	epochs=5
	dropout=0.0
	batch_size=128
	num_hidden=1
	hidden_layer_size=128
else:
	bin_width = 50000
	model_save_file = 'model_saved.h5'
	epochs = 5
	dropout = 0.1
	batch_size = 128
	num_hidden=1
	hidden_layer_size=128


test_prices = np.load(data_file_prefix+'_test_Y.npy')

raw_predictions, history = run_instance((X_train, Y_train, X_test, Y_test), epochs, dropout, batch_size, num_hidden, hidden_layer_size, model_save_file)
predicted_prices = get_price(raw_predictions, test_prices, bin_width)
conservative_preds = conservative_predict(raw_predictions, test_prices, bin_width)

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.title("Training and Validation Set Accuracy Versus Epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
acc_h, = plt.plot(range(1, epochs+1), acc, label='Training Set Accuracy')
val_acc_h, = plt.plot(range(1, epochs+1), val_acc, label='Validation Set Accuracy')
plt.legend(handles=[acc_h, val_acc_h])
plt.savefig("accuracy_v_epochs.png")
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.title("Training and Validation Set Loss Versus Epochs")
plt.ylabel("Loss")
plt.xlabel("Epochs")
loss_h, = plt.plot(range(1, epochs+1), loss, label='Training Set Loss')
val_loss_h, = plt.plot(range(1, epochs+1), val_loss, label='Validation Set Loss')
plt.legend(handles=[loss_h, val_loss_h])
plt.savefig("loss_v_epochs.png")
plt.show()