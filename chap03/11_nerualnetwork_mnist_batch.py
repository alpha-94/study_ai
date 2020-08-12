import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return (x_test, t_test)

def init_network():
    with open("sample_weight.pkl","rb") as file:
        network = pickle.load(file)

    return network


def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'],network['W2'], network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x, W1) + b1 #X는 100*784 행렬.
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    batch_size = 100 # 배치 크기

    for i in range(0,len(x),batch_size): # 처음 0, 100, 200 순으로 100단위로 반복. 한번에 읽어올 때 100개씩 읽어와 처리하겠다는 의지치. 100번만 반복.
        x_batch = x[i:i+batch_size] # 100개 읽어옴. 100 x 784 행렬 형태 -> 선형대수를 사용. 한번에 처리 가능.
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        '''
        x = np.array[[0.1, 0.8, 0.1],[0.3, 0.1, 0.6],
                     [0.2, 0.5, 0.3],[0.8, 0.1, 0.1]])
        y = np.argmax(x, axis=1)
        print(y) # [1 2 1 0]
        '''
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

        '''
        y = np.array([1,2,1,0])
        t = np.array([1,2,0,0])
        print(y==t) # [True True False True]
        print(np.sum(y==t)) # 3
        '''

    print("Accuracy : " + str(float(accuracy_cnt)/len(x)))





