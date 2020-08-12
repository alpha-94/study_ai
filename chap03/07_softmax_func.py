import numpy as np

# 최종 결과물이 3가지로 분류되는 신경망 가정.
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) #배열값을 넣어주면 각각 계산됨 #exp() 함수는 밑(base)이 자연상수 e 인 지수함수 로 변환
print(exp_a) # [ 1.34985881 18.17414537 54.59815003] 소프트맥스 수식에서의 분자

sum_exp_a = np.sum(exp_a) #소프트맥스함수의 분모.
print(sum_exp_a) #74.1221542101633

y = exp_a / sum_exp_a # 배열 값을 각각 계산해줌
print(y) #[0.01821127 0.24519181 0.73659691] 입력한 3가지 데이터의 비율값 출력


def softMax_func(x):
    return np.exp(x)/np.sum(np.exp(x)) # exp 함수를 적용한 모든 합을 exp 적용 값을 나눈 값

print(softMax_func(a))