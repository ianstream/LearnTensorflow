#-*- coding: utf-8 -*-

import tensorflow as tf

# tf graph input
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# construct linear model
hypothesis = W * X

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient descent algorithm
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

# initilize the variables. we will 'run' this first
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(100):
    # 프로그램이 실행되는 시점에 값이 지정됨
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

