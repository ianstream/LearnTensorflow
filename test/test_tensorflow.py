import tensorflow as tf

hello = tf.constant('Hello, tensorflow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(20)

# run 을 하기 전에는 실제 연산이 되는 것이 아님
print(sess.run(a + b))
