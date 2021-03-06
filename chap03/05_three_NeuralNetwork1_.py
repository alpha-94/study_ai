import numpy as np

#1층에서의 활성화 함수 처리
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 항등 함수의 정의 - 출력단
def identity_function(x):
    return x


# 입력층에서 1층으로의 신호 전달
# 선형대수? 행렬를 의미. 데이터를 행과 열로 구성하여 연산을 빠르게 수행하는 것.
X = np.array([1.0, 0.5]) #입력은 두개, 출력은 3개. -> W는 2X3 (6개) 구성 필요
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1, 0.2, 0.3]) #bias(편향) 3개 : 출력이 3개이기 때문에

print(W1.shape) # (2, 3)
print(X.shape)  # (2,)
print(B1.shape) # (3,)

# 신호를 a1에게 전달. a1은 어떻게 전달 받으면 될까?
# 매트릭스의 경우 대문자로 표기. 넘파이의 도트함수를 이용하여 입력신호와 가중치의 내적을 계산
# 각 층의 신호 전달 구현 – 입력층에서 1층으로 신호 전달
A1 = np.dot(X,W1) + B1 #결과 a(1). 여기서 활성화 함수(시그모이드)에 넣어 나오는 데이터를 다시 X에 넣어서 맵핑되는 Y에 출력, 다음 노드에 전달
Z1 = sigmoid(A1)  #활성화 함수 시그모이드
# 넘파이는 배열 데이터를 전송되어져 오면 배열로 계산되도록 최적화.
# Z1의 결과값은 3개.

print(A1) #[0.3 0.7 1.1] -> 임의로 집어 넣은 1, 0.5 에 가중치를 계산한 결과. 이후 활성화 함수 h(x) [시그모이드 함수] 적용
print(Z1) #[0.57444252 0.66818777 0.75026011] -> A1에 활성화 함수 적용. 첫번째 은닉층의 결과.
# A1에서 활성화함수를 거쳤더니, 1을 넘어가는 데이터를 포함 모든 값은 0초과 1미만의 값으로 출력.
# 시그노이드 그래프에서 x가 0.3 일 때 y는 0.57444252 / 0.7일 땐, 0.66818777 / 1.1일 땐 0.75026011

print()
# 1층에서 2층으로의 신호 전달
# 1층 결과물(Z1)이 다시 입력 데이터가 되어 출력하는 시스템.
# 입력 3개 , 출력 2개  -> W는 3행 2열이 되어야 함.
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2]) #bias(편향) 2개. 0.2가 0.1보다 신호의 출력을 내보내기 어렵게 만들어 놈.

A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)


# 2층에서 3층(출력층)으로의 신호 전달
# 마지막 출력 신호를 최종 출력단에서 입력 신호로 작용.
# 입력은 2개 출력은 2개 -> 가중치 2행 2열
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2,W3) + B3

# 항등함수? 입력을 그대로 출력, 회귀. 버퍼기능. 잠깐 보관하고 있다가 한번에 내보내는 기능
# 머신러닝의 학습 방법? 지도학습, 비지도학습. 지도학습의 대표적 알고리즘은 회귀, 분류.
y = identity_function(A3) # 항등함수 적용
print(A3) #[0.31682708 0.69627909]
print(y) #[0.31682708 0.69627909]

