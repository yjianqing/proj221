from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--learning_rate", type=float, default=0.1)
parser.add_argument("-s", "--num_steps", type=int, default=500)
parser.add_argument("-b", "--batch_size", type=int, default=512)
parser.add_argument("-l", "--layers", nargs="*", type=int, help="Nodes in each hidden layer e.g. for 2 layers of 256 nodes each: 256 256")
parser.add_argument("-o", "--output_file", default="model.ckpt", help="Output .ckpt filename")
parser.add_argument("-g", "--use_gpu_options", action="store_true")
parser.add_argument("-d", "--display_step", type=int, default=10)
args = parser.parse_args()

learning_rate = args.learning_rate
num_steps = args.num_steps
batch_size = args.batch_size
display_step = args.display_step

gpu_options = None
if args.use_gpu_options:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

n_hidden = args.layers
if n_hidden is None:
    n_hidden = [256, 256]

print("Settings:")
print("Learning rate: ", learning_rate)
print("Steps: ", num_steps)
print("Batch size: ", batch_size)
print("Hidden layer nodes: ", n_hidden)

trainX = np.load('hdb_train_X.npy')
trainY = np.load('hdb_train_Y.npy')
trainX, trainY = unison_shuffle(trainX, trainY)
testX = np.load('hdb_test_X.npy')
testY = np.load('hdb_test_Y.npy')
testX, testY = unison_shuffle(testX, testY)

num_input = trainX.shape[1]
num_examples = trainX.shape[0]
test_size = testX.shape[0]
num_classes = 1

trainX = tf.convert_to_tensor(trainX, np.float32)
trainY = tf.convert_to_tensor(trainY, np.float32)
testX = tf.convert_to_tensor(testX, np.float32)
testY = tf.convert_to_tensor(testY, np.float32)

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = dict()
biases = dict()
for layer in range(len(n_hidden) + 1):
    label = 'h' + str(layer+1)
    innodes = None
    outnodes = None
    if layer == 0:
        innodes = num_input
    else:
        innodes = n_hidden[layer-1]
    if layer == len(n_hidden):
        label = 'out'
        outnodes = num_classes
    else:
        outnodes = n_hidden[layer]
    weights[label] = tf.Variable(tf.random_normal([innodes, outnodes]))
    biases[label] = tf.Variable(tf.random_normal([outnodes]))

def neural_net(x):
    layer_n = x
    for layer in range(len(n_hidden)):
        layer_n = tf.nn.relu(tf.add(tf.matmul(layer_n, weights['h'+str(layer+1)]), biases['h'+str(layer+1)]))
    out_layer = tf.matmul(layer_n, weights['out']) + biases['out']
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
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

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
    saver = tf.train.Saver()
    saver.save(sess, "./" + args.output_file)
    
    i = 0
    while i < test_size:
        batch_x = testX[i:i+batch_size]
        batch_y = testY[i:i+batch_size]
        i += batch_size
        print(sess.run(loss_op, feed_dict={X: batch_x.eval(), Y: np.expand_dims(batch_y.eval(), axis=1)}))
