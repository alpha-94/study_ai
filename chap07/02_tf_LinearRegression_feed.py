import  tensorflow as tf

tf.set_random_seed(777)


W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# 가설함수 H(x) 정의
hypothesis = X * W + b

# 손실(loss, cost)함수 정의
loss = tf.reduce_mean(tf.square(hypothesis - Y))


# 경사 하강법(Gradient Descent algorithm)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(2001) :
    loss_val , w_val , b_val , _ = session.run([loss,W,b,train],feed_dict={X:[1,2,3],Y:[1,2,3]})

    if step % 20 ==0 :
        print(step, loss_val, w_val, b_val)

print(session.run(hypothesis, feed_dict={X:[5]}))
print(session.run(hypothesis, feed_dict={X:[2.5]}))
print(session.run(hypothesis, feed_dict={X:[1.5, 3.5]}))
