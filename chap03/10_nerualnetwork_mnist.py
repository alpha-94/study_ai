import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data(): #이미지 읽어오는 함수
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return (x_test, t_test) #테스트 데이터 리턴

def init_network(): # 학습시킨 모델 정보를 담아줌. #sample_weight.pkl 다운. 미리 만들어논 모델
    with open("sample_weight.pkl","rb") as file:
        network = pickle.load(file) #pickle 모듈 . 파이썬 제공하는 자료형 데이터. pkl포맷. 파이썬이 데이터를 빠르게 처리 수행할 수 있는 자료형. 신경망에 대한 정의됨

    return network #신경망 리턴


def sigmoid(x): #시그모리드 활성화 함수
    return 1 / ( 1 + np.exp(-x))

def softmax(a): #소프트맥스- 분류알고리즘 최출력단에 연결
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x): #파일 저장된 모델 네트워크의 가중치, 편향 값을 꺼내옴. 꺼내와서
    W1, W2, W3 = network['W1'],network['W2'], network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    print(W1.shape) #신경망 형태 확인
    print(W2.shape)
    print(W3.shape)

    """ 출력값
    (784, 50) -> 1층 은닉층 : 입력 784개. 노드 50개
    (50, 100) -> 2층 은닉층 : 입력 50개, 노드 100개
    (100, 10) -> 최종 출력 10개. 0~9까지 분류되기 때문에.
    """

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#모델이 이미 생성. 모델의 정확도를 체크
if __name__ == "__main__":
    x, t = get_data() #x는 이미지 만개. t 레이블(정답)
    network = init_network() #공유한 파일을 읽어와 네트워크에 대한 정보를 리턴.

    accuracy_cnt = 0 #변수 선언 초기화

    for i in range(len(x)): # 이미지 테스트 만번 반복
        y = predict(network, x[i]) #첫번째 이미지를 신경망에 전달. ->은닉층을 거쳐 최종 출력 후 y에 반환
        p = np.argmax(y)

        # 신경망의 입력 갯수는 784개를 입력 받도록 설정. 이미지 크기가 28*28이고 펼치면 784 픽셀.
        # 은닉층을을 거쳐 분류알고리즘을 적용. 숫자 이미지 8이 들어오면 8이라고 예측을 기대.
        # 분류 ? 입력 데이터를 분류하는 것. mnist는 0~9까지 사람이 쓴 손글씨. 이미지로 저장된 것. 분류 알고리즘을 놓고 보면 10가지 결과를 예측해야 함.
        # 입력은 784개 최종 출력 10개 -> 은닉층 2층(시그모이드2개)-> 최종결과 실수값/1미만 , 최종 결과 1층(소프트맥스1개) -> 비율이 결과값
        # argmax함수? argument max. 소프트맥스로 출력되는 비율 중 최고 큰 값을 찾아 인덱스를 반환. 레이블이 인덱스로 되어 있기 때문. 인덱스가 8이면 숫자 8을 반환.

        if p == t[i]: #인덱스 == 레이블 비교.
            accuracy_cnt += 1 #만번 비교. 10000이 출력되면 100% 맞춘 것.


    print("Accuracy : " + str(float(accuracy_cnt)/len(x))) #정확도 = 맞춘 갯수/전체 갯수
    # Accuracy : 0.9352. 한번도 보지 못한 이미지를 93% 정확도로 맞춤.
    # 노드를 증감할 수록 정확도를 변경됨.
