import tensorflow as tf

tf.set_random_seed(777)

# matrix

score = [[73.,80.,75.],
         [93.,88.,93.],
         [89.,91.,90.],
         [96.,98.,100.],
         [73.,66.,70.]]

final_term = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설함수 정의 (Y = XW + B)
hypothesis = tf.matmul(X,W) + b

# 손실함수 정의 (MSE)
loss = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 알고리즘
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001) : # 학습을 늘리면 정확도는 높아지지만 많이 늘리게 되면 오버헤딩 발생 가능성 있음
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={X : score , Y : final_term})

    if step % 100==0:
        print(step, 'Loss : ', loss_val, '\n 예측값 : \n', hy_val)



print('test-set : ', sess.run(hypothesis, feed_dict={X : [[75.,70.,72.]]}))























