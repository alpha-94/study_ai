import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# 당뇨병 환자 데이터 셋
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
print(xy)
