#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print(x_data)
print(y_data)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0)) # 2차원으로 제공
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# hypothesis
hypothesis = tf.matmul(W, x_data) # 행렬 곱셈

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# initilize the variables. we will 'run' this first
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))

