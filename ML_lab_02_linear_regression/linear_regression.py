import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]

"""
try to find vlaues for W and b compute y_data = W * x_data + b
(we know that W should be 1 and b 0, but Tensorflow will figure that out for us)
"""
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# our hypothesis
hypothesis = W * x_data + b

# simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initilize the variables. we will 'run' this first
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
