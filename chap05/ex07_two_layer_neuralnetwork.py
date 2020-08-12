import numpy as np
from common.functions import * #모듈에 정의되어진 모든 기능
from common.gradient import numerical_gradient
from collections import OrderedDict #파이썬이 제공하는 collections 내 OrderedDict함수 -> 딕셔너리 자료를 순서를 가지고 관리하는 함수
from common.layers import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0):
        #가중치 초기화. 레이어 2층
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #randn 정규분포를 따르고 있는 데이터셋에서 데이터를 랜덤하게 꺼내오겠다는 것
        self.param['b1'] = np.zeros(hidden_size) #hidden_size만큼 shape만든 후 0으로 담아줌
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict() #OrderedDict 딕셔너리형태로 순서 유지. 키-벨류형태로 저장 가능
        self.layers['Affine1'] = Affine(self.param['W1'], self.param['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.param['W2'], self.param['b2']) #score = 계산되어 나오는 결과값. -> 소프트맥스 -> 손실함수
        self.lastLayer = SoftmaxWithLoss()


    # 데이터에 대한 흐름. 2층 정리
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 손실함수 정의
    # x : 입력데이터, t : 정답레이블
    def loss(self, x, t):
        y = self.predict(x) #predict 함수를 실행해서 소프트맥스를 거친 최종 결과물이 나옴

        return self.lastLayer.forward(y, t) # y는 예측값. t는 정답. SoftmaxWithLoss()


    def accuracy(self, x, t): # accuracy 정확도를 얼만큼 가져가는지 체크. 정확도 계산의 함수를 정의
        y = self.predict(x) #우선 입력 데이터를 학습시킴. 이후 결과를 y 를 저장
        y = np.argmax(y, axis=1) # 계산되어진 최종 값의 가장 큰 값을 찾아, 인덱스를 리턴.
        if t.ndim != 1 :
            t = np.argmax(t,axis=1)


        # 꺼내온 y, t 인덱스가 동일할 때 sum하여 전체 데이터를 나눔
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 경사하강법 적용을 위해 편미분값을 구해야 함.
    # x : 입력데이터, t : 정답레이블
    """
    def loss_W(W):
    return self.loss(x,t)
    """

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t) #lambda ? 코드를 간편하게 정의할 수있는 기능. 아래 함수를 한줄로 요약

        grads = {} #자료형 딕셔너리로 담겠다는 의지치
        # common 밑에 numerical_gradient 함수 호출. 같은 클래스 안에 있는 numerical_gradient 메소드와 이름이 동일한데 가까운 내 것을 호출하는 것은 아닐까?
        # 클래스 안에 정의한 함수나 필드는 그 자료형안에서 호출하고 싶으면, self를 붙여야 함. self가 안붙여져 있기에 외부에 있는 함수를 불러오는 거라고 정의/
        # numerical_gradient는 중앙정렬로 정의된 미분을 구하는 함수. 매트릭스 안을 편미분하면서 독립적인 기울기를 구함.
        # 최종적으로 보정된 값으로 각각 적용.
        grads['W1'] = numerical_gradient(loss_W, self.param['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.param['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.param['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.param['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1 #초기화 설정. 생략 가능
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db #역전파에 의해 출력되어진 dW을 꺼내와 저장
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return  grads












