import numpy as np
# Loss Function

# - MSC(Mean Squared Error)
def mean_squared_error(t , y): # t = 정답 / y = 예측
    return np.sum((y - t) ** 2) # 하나의 값만 계산



# - CEE(Cross Entropy Error)
def cross_entropy_error(t , y):
    delta = 1e-7 # nan 보정값
    return -np.sum(t*np.log(y+delta))

# 정답 :: 2
t = [0,0,1,0,0,0,0,0,0,0] # one-hot encoding

# mean_squared_error
# 예측 결과 : 2
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
msq = mean_squared_error(np.array(t), np.array(y))
print(msq) # 0.19500000000000006 # 면적의 평균 = 0으로 만들어 줘야함

# 예측 결과 : 7
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
msq = mean_squared_error(np.array(t), np.array(y))
print(msq) # 1.195

# cross_entropy_error
# 예측 결과 : 2
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
cee = cross_entropy_error(np.array(t), np.array(y))
print(cee) # 0.510825457099338

# 예측 결과 : 7
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
cee = cross_entropy_error(np.array(t), np.array(y))
print(cee) # 2.302584092994546