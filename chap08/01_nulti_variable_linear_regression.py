import tensorflow as tf

tf.set_random_seed(777)

quiz1 = [73., 93., 89., 96., 73.]
quiz2 = [80., 88., 91., 98., 66.]
midterm = [75., 93., 90., 100., 70.]

finalterm = [152., 185., 180., 196., 142.] # 예측모델

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32) # 예측값

w1 = tf.Variable(tf.random_normal([1]), name='weight1') # 가변
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')

b = tf.Variable(tf.random_normal([1]), name='bias')


# 가설함수 정의 (Y = XW + B)
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# 손실함수 정의 (MSE)
loss = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 알고리즘
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001) : # 학습을 늘리면 정확도는 높아지지만 많이 늘리게 되면 오버헤딩 발생 가능성 있음
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x1 : quiz1, x2 : quiz2, x3 : midterm, Y : finalterm})

    if step % 100==0:
        print(step, 'Loss : ', loss_val, '\n 예측값 : \n', hy_val)

# 10000 적용 할 때 [150.93231 184.98941 180.49393 196.34583 142.05391] # Loss :  0.301307
# 15000 적용 할 때 [151.22945 184.79118 180.59178 196.36273 141.8395 ] # Loss :  0.22897995
# 20000 적용 할 때 [151.31877 184.73473 180.62505 196.34004 141.80513] # Loss :  0.2157456