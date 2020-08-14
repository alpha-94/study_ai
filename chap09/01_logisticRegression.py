import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] # [수업시간,개인시간]
y_data = [[0],[0],[0],[1],[1],[1]] # [합격/불합격]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# 가설함수 :: Logistic Regression
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# 손실함수(CEE) :: -ylog(H(x)) - (1-y)log(1-H(x))
loss = - tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))


# 경사하강법
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimize.minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 임계치
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y),dtype=tf.float32)) # 정확률

with tf.Session() as sess: # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _ , loss_val= sess.run([train, loss], feed_dict={X : x_data, Y : y_data})
        if step % 200 ==0 :
            print('step :: ', step ,'손실 값 :: ' , loss_val)

    h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={X : x_data, Y : y_data})
    print('hypothesis :: ',h,'predict :: ',p,'accuracy :: ',a)
    print('test-set : ', sess.run(predict, feed_dict={X : [[3,4]]}))


sess.close()
