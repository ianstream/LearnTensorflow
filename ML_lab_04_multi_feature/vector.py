#-*- coding: utf-8 -*-

import tensorflow as tf

# tf graph input
x_data = [[0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]

y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0)) # 2차원으로 제공
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# hypothesis
hypothesis = tf.matmul(W, x_data) + b # 행렬 곱셈

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
        print(step, sess.run(cost), sess.run(W), sess.run(b))

