import numpy as np
import matplotlib.pylab as plt

# 미분의 나쁜 구현 예 (문제점 2가지)
def numerical_diff_error(f,x):

    h = 10e-50  # 0에 가깝게 처리
                # 문제점 1 :: python의 경우 np.float32(1e-50)는 0.0 으로 처리(반올림 오차)
                # 문제점 2 :: h에 의한 오차 발생
    return (f(x+h) - f(x)) / h

# 문제점 해결 수치 미분의 예
def numerical_diff(f , x):
    h = 1e-4 # 0.0001

    return (f(x + h) - f(x - h)) / 2*h  # x 를 중심으로 그 전후의 차분을 계산한다. :: 중심(중앙) 차분


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("X")
plt.ylabel("F(X)")
plt.plot(x , y)
# plt.show()

print(numerical_diff(function_1,5)) # 1.9999999999908982e-09
print(numerical_diff(function_1,10)) # 2.999999999986347e-09

# 편미분의 예
# -f(x0,x1) = x0**2 + x1**2

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x ** 2)

# x0 = 3, x1 = 4 일때 , x0 에 편미분
def function_tmp1(x0): # x0 에 편미분
    return x0 * x0 + 4 ** 2

# x0 = 3, x1 = 4 일때 , x1 에 편미분
def function_tmp2(x1): # x1 에 편미분
    return 3 ** 2 + x1 * x1

print(numerical_diff(function_tmp1, 3)) # 수치 미분값 : 6.000000000003781e-08

print(numerical_diff(function_tmp2, 4)) # 수치 미분값 : 7.999999999999119e-08













































