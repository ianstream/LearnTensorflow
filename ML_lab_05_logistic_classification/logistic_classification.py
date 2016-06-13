#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# # 시군구,전용면적(㎡),거래금액(만원),건축년도
# xy = np.genfromtxt('test.csv', delimiter=',', unpack=True, dtype='float32')
xy = np.loadtxt('test.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# minimize
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


print("===============================================")

print(sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5)

print(sess.run(hypothesis, feed_dict={X:[[1, 1], [4, 3], [3, 5]]}) > 0.5)
