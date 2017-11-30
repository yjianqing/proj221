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

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

data_file_prefix = '../data_process/hdb'

X_train = np.load(data_file_prefix+'_train_X.npy')
Y_train = np.load(data_file_prefix+'_binned_train_Y.npy')
X_train, Y_train = unison_shuffle(X_train, Y_train)

X_test = np.load(data_file_prefix+'_test_X.npy')
Y_test = np.load(data_file_prefix+'_binned_test_Y.npy')

num_features = X_train.shape[1]
num_classes = np.max(Y_train) + 1
assert num_classes > np.max(Y_test)

# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


def baseline_model():
	hidden_layer_size = 512

	model = Sequential()
	model.add(Dense(hidden_layer_size, activation='relu', input_shape=(num_features,)))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_layer_size, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])	
	return model

batch_size = 512
epochs = 10
model = baseline_model()
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


raw_predictions = model.predict(X_test)

def get_price(raw_predictions, bin_width = 20000):
	bin_prices = np.array(range(num_classes))*bin_width
	prices = np.dot(raw_predictions, bin_prices)
	return prices

predicted_prices = get_price(raw_predictions)
test_prices = np.load(data_file_prefix+'_test_Y.npy')
rms = np.sqrt(np.mean(np.square(predicted_prices-test_prices)))
print("RMS on test data: {}".format(rms))

mean_abs = np.mean(np.absolute(predicted_prices-test_prices))
print("Mean absolute on test data: {}".format(mean_abs))

model.save('cached_NN.h5')
