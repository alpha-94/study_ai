import numpy as np

# Vector(1차원 배열)
X = np.array([1,2,3,4,5])
print(X.shape) # (5,)

# 3x2 행렬과 2x3 행렬의 내적 (matrix :2 차원 배열)
A = np.array([[1,2],[3,4],[5,6]])
print(A.shape) # (3, 2)
B = np.array([[1,2,3],[4,5,6]])
print(B.shape) # (2, 3)

Z = np.dot(A,B)
print(Z)
'''
[[ 9 12 15]
 [19 26 33]
 [29 40 51]]

'''
print(Z.shape) # (3, 3)

# Array(3차원 배열 이상) - 2면 3행 4열
X = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
              [[13,14,15,16],[17,18,19,20],[21,22,23,24]]])

print(X.shape) # (2, 3, 4)
print(np.ndim(X)) # 3

# 교환법칙 성립 X
Z2 = np.dot(B,A)
print(Z2)
'''
[[22 28]
 [49 64]]
'''
