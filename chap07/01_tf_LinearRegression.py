# tensorflow 를 이용한 Linear Regression(선형회귀)

import tensorflow as tf

tf.set_random_seed(777)

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

# 가설 : y = wx + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

# 손실(loss, cost)함수 정의
loss = tf.reduce_mean(tf.square(hypothesis - y_train))


# 경사 하강법(Gradient Descent algorithm)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(4001) :
    session.run(train)

    if step % 20 ==0 :
        print(step, session.run(loss), session.run(W), session.run(b))
