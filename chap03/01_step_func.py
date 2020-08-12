# 계단 함수(Step Function)

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    if x >0:
        return 1
    else:
        return 0


def step_func_ndarray(x):
    y = x > 0
    return y.astype(np.int)


if __name__ == '__main__':
    x = step_function(3)
    print(x) # 1

    x = step_function(-3)
    print(x) # 0

    z = np.array([-1,1,2])

    # x = step_function(z)
    #print(x)  # error
    print(step_func_ndarray(z))


    x = np.arange(-5,5,0.1)
    y = step_func_ndarray(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()