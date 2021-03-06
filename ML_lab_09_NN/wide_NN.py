# xor with neural network
import numpy as np
import tensorflow as tf

xy = np.loadtxt('dataset.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name='Weight2')

b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Hypotheses
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1)+b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1.-hypothesis))
    cost_sum = tf.scalar_summary('cost', cost)

# Minimize cost.
with tf.name_scope("train") as scope:
    a = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# add histogram
w1_hist = tf.histogram_summary('weights1', W1)
w2_hist = tf.histogram_summary('weights2', W2)

b1_hist = tf.histogram_summary('biases1', b1)
b2_hist = tf.histogram_summary('biases2', b2)

y_hist = tf.histogram_summary('y', Y)


# Initializa all variables.
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    with tf.name_scope("accuracy") as scope:
        # Test model
        correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_sum = tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/xor_logs', sess.graph_def)

    for step in range(8001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print(
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                sess.run(W1),
                sess.run(W2),
                accuracy
            )

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))