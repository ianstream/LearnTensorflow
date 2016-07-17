# xor with logistic regression

import tensorflow as tf
import numpy as np


xy = np.loadtxt('dataset.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

print(x_data)
print(y_data)

X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-input')

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0), name='weight')

# hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# cost fucntion
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# minimize
a = tf.Variable(0.01)  # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize variables

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # fit the line
    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    # test model
    # hypothesis : 0~1. 0.5 를 넘으면 1로 나와야 함
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # calcurate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
    print("accuracy", accuracy.eval({X: x_data, Y: y_data}))
