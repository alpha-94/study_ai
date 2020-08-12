import numpy as np
import matplotlib.pylab as plt

def numerical_gradient_no_batch(f,x):
    h = 1e-4
    grad = np.zeros_like(x) # x 와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h)계산
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        # 중앙차분
        grad[idx] = (fxh1-fxh2) / (2 + h)
        
        #값 복원
        x[idx] = tmp_val

    return grad

def numerical_gradient(f,x):
    if x.ndim == 1:
        return numerical_gradient_no_batch(f,x)
    else:
        grad = np.zeros_like(x)
        for idx, z in enumerate(x):
            grad[idx] = numerical_gradient_no_batch(f,x)

        return grad


def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    x_history = [] # 학습할때마다 결과치 나온 것 저장위치

    for i in range(step_num):
        x_history.append(x.copy())

        # 편미분
        grad = numerical_gradient(f,x)
        # 경사하강 누적
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x): # f(x0,x1) = x0^2 + x1^2
    return x[0] **2 + x[1]**2

if __name__ == '__main__':
    init_x = np.array([-3.0,4.0])

    lr = 0.1 # learning rate
    step_num = 20

    x, x_history = gradient_descent(function_2, init_x, lr, step_num)

    plt.plot([-5, 5], [0,0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5,3.5)
    plt.ylim(-4.5,4.5)

    plt.xlabel('x0')
    plt.ylabel('x1')

    plt.show()