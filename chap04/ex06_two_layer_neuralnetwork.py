import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet: # 신경망 :: 입력층 1 은닉층 1 출력층 1
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01): # hidden_size :: 은닉층 사이즈
        # 가중치 초기화
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # randn :: nomalize
        self.param['B1'] = np.zeros(hidden_size) # zeros(shape_size)
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['B2'] =np.zeros(output_size)



    def pridict(self, x): # 흐름
        W1, W2 = self.param['W1'], self.param['W2']
        B1, B2 = self.param['B1'], self.param['B2']

        a1 = np.dot(x,W1) + B1 # 행렬곱 (내적)
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + B2 # 행렬곱 (내적)
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.pridict(x) # 예측값

        return cross_entropy_error(y, t) # 오차가 최소화가 되어지는 값으로 보정

    # 정확도 체크
    def accuracy(self, x, t):
        y = self.pridict(x)  # 예측값
        y = np.argmax(y,axis=1) # 인덱스 리턴 // axis = 1 : 열단위
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터 , t : 정답 레이블

    '''
   lambda 의미 :: def 정의하지 않고 변수에 바로 지정할 때 쓰임 

   def loss_W(W):
       return self.loss(x,t)
   '''

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)



        grads = {}

        # 클래스 내 자가 호출 시 self 를 해야하기 때문에 역으로 보면 같은 이름이라도 self 로 호출하지 않는한 자신의 함수가 아닌걸로
        # 판단하기 때문에 numerical_gradient 함수는 외부 함수로 인식 한다.
        grads['W1'] = numerical_gradient(loss_W, self.param['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.param['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.param['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.param['B2'])

        return grads






















