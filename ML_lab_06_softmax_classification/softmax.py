#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# tf graph input
X = tf.placeholder("float", [None, 3]) # x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None, 3]) # A, B, C ==> 3 classes

# set model weights
W = tf.Variable(tf.zeros([3, 3]))

# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))  # softmax

# minimize error using cross entropy
learning_rate = 0.001

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    print("===============================================")
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(all, sess.run(tf.arg_max(all, 1)))


"""
[[ 0.46272627  0.35483009  0.18244369]] [0] -> a 가 될 확률 46% 로 해석, 나머지도 동일하게 해석하면 됨
[[ 0.33820099  0.42101383  0.24078514]] [1]
[[ 0.27002314  0.29085544  0.4391214 ]] [2]
[[ 0.46272627  0.35483006  0.18244369]
 [ 0.33820099  0.42101383  0.24078514]
 [ 0.27002314  0.29085544  0.4391214 ]] [0 1 2]

"""
