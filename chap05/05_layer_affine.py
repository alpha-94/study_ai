import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x
        out = np.dot(x,self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #W.T : W에 대한 전치행렬
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) #각각의 행을 기준으로 열의 합을 구함. 최종 3개의 데이터만 남게 됨.

if __name__ == "__main__":
    X_dot_W = np.array([[0,0,0],[10,10,10]])
    B = np.array([1,2,3])

    print(X_dot_W + B)

    print()
    dY = np.array([[1,2,3],[11,12,13]])
    dB = np.sum(dY, axis=0) #axis 축. 1차원 = 0, 2차원 = 0,1 / 지금은 2차원. 행단위로 데이터를 바라보며 열단위의 데이터의 합을 계산
    print(dB)



