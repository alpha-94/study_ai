import  numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a= np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

#오버플로우 처리
def softmax_computer(a):
    max = np.max(a)
    exp_a = np.exp(a-max) #오버플로우 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y) #[0.01821127 0.24519181 0.73659691]

# 결과값이 클 때?  오버플로우 문제 발생.
# 시그모이드를 적용하면 비선형을 가져가지만 1미만으로 출력되기 때문에 입력 신호의 크기를 상쇄해버림. 해서 최근 ReLu함수를 많이 사용
# ReLu함수는 양수 입력값에 대해 그대로 출력. 예측분석의 활성화 함수로 적용하면 더 높은 정확도를 가짐. 해서 아래와 같이 큰 값이 결과값으로 나올 가능성 높음
a = np.array([1010,1000,990])
"""
y = softmax(a)
print(y) #[nan nan nan] not a number
"""
# 지수함수? x값이 음수이면 y가 0에 가깝고 양수이면 y가 급격하게 커짐.
# e1010 = 2.715의 1010승 으로 급격하게 크기 때문에 y역시 급격하게 큼. -> nan / RuntimeWarning: overflow encountered in exp

max = np.max(a) # a값에서 최대값
result = np.exp(a-max) / np.sum(np.exp(a-max)) #-max를 빼면 소프트맥스 함수
print(result) #[9.99954600e-01 4.53978686e-05 2.06106005e-09] : 정상적으로 출력

y = softmax_computer(a)
print(y) #[9.99954600e-01 4.53978686e-05 2.06106005e-09]
