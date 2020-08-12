import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train) , (x_test,t_test) =\
    load_mnist(normalize=True, one_hot_label=True) # 정규화 T / 원 핫 인코딩 방식 적용 T


print(x_test.shape) # (10000, 784)
print(t_train.shape) # (60000, 10)

print(t_train.shape[0]) # 60000

train_size = t_train.shape[0]
batch_size = 100 # 무작위 뽑기

# 지정한 범위 수 중에서 무작위로 원하는 개수만 선택 .
batch_mask = np.random.choice(train_size, batch_size) # 전체범위 , 미니배치 범위 -> index 반환
print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]





















