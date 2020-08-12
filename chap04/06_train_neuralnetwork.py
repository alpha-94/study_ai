import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from ex06_two_layer_neuralnetwork import TwoLayerNet

# 데이터 읽기
(x_train,t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784 , hidden_size =50 , output_size = 10) # hidden_size 는 튜닝값

# 하이퍼파라미터
learning_rate = 0.1
train_size = x_train.shape[0] # 60000
batch_size = 100 # 배치 크기 : 100개씩 가져오기
iters_num = 10000 # 반복 횟수

train_loss_list = [] # 손실값이 작아지는지 확인하기 위한 리스트변수 (시각화)
train_acc_list = []
test_acc_list = []

# 1 에폭당 반복 수 : 1에폭 = train_size / batch_size = 600번
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size) # 0~59999 값에서 100개를 랜덤으로 뽑기
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산 (미분)
    grad = network.numerical_gradient(x_batch,t_batch)

    # 매개변수(가중치, 편향) 갱신
    for key in ('W1','B1','W2','B2'):
        network.param[key] -= learning_rate * grad[key]
        
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch 당 정확도 계산
    if i % iter_per_epoch ==0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | '+str(train_acc)+',' + str(test_acc))

# 그래프 그리기
markers = {'train' : 'o', 'test' : 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label = 'train acc')
plt.plot(x, test_acc_list, lable='test acc',linesstyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1.0)
plt.legend(loc = 'lower right')
plt.show()