import numpy as np
from common.util import im2col

class Convolution:
    def __init__(self, W,b,stride =1, pad=0):
        self.W = W # 필터
        self.b = b #
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(((H+2 * self.pad -FH) / self.stride) + 1) # 이미지 행 (높이)
        out_w = int(((H+2 * self.pad -FW) / self.stride) + 1) # 이미지 열 (너비)

        col = im2col(x, FH, FW, self.stride, self.pad) # 4차원 -> 2차원

        col_W = self.W.reshape(FN,-1).T # -1 : 가변값  / im2col 은 10X~ reshape.T 는 ~X10

        out = np.dot(col, col_W) + self.b
        out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2) # transpose(0,3,1,2) :: 데이터 개수[0] / 채널[3] / 행[1] / 열[2]

        return out


    def backward(self):
        pass