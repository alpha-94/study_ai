import numpy as np
from common.util import im2col


x = np.random.rand(10,1,28,28) # 4차원 데이터 입력 쉐입(개당 배치 이미지, 채널수, 데이터 행, 데이터 열)
print(x[0].shape) # (1, 28, 28)
print(x[0,0].shape)  # (28, 28)

# print(x[0,0])

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5) # 2차원
print(col1.shape) #(9, 75)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5) # 2차원
print(col2.shape) #(90, 75)