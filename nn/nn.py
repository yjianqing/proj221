from __future__ import print_function
import numpy as np
import tensorflow as tf

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

learning_rate = 0.1
num_steps = 500
batch_size = 512
display_step = 5

n_hidden_1 = 256
n_hidden_2 = 256

trainX = np.load('hdbTrain_X.npy')
trainY = np.load('hdbTrain_Y.npy')
trainX, trainY = unison_shuffle(trainX, trainY)
testX = np.load('hdbTest_X.npy')
testY = np.load('hdbTest_Y.npy')
testX, testY = unison_shuffle(testX, testY)

num_input = trainX.shape[1]
num_examples = trainX.shape[0]

trainX = tf.convert_to_tensor(trainX, np.float32)
trainY = tf.convert_to_tensor(trainY, np.float32)
testX = tf.convert_to_tensor(testX, np.float32)
testY = tf.convert_to_tensor(testY, np.float32)

num_classes = 1

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, trainY))))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(num_steps):
        batch_x = trainX[(batch_size*step)%num_examples:(batch_size*(step+1))%num_examples]
        batch_y = trainY[(batch_size*step)%num_examples:(batch_size*(step+1))%num_examples]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x.eval(), Y: np.expand_dims(batch_y.eval(), axis=1)})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run(loss_op, feed_dict={X: batch_x.eval(), Y: np.expand_dims(batch_y.eval(), axis=1)})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    #print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
