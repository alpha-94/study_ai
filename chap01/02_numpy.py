import numpy as np
import matplotlib.pyplot as plt
# 3.7 버전은 matplotlib 패키지가 자동으로 패키지 구현이 안되어있어서 수동으로 설치 해야함

# 배열생성

# - 1차원 배열(Vector) 정의
arr = np.array([1,2,3])
print(arr) # [1 2 3]
print(type(arr)) # <class 'numpy.ndarray'> :: n 차원 dimension = nd + array
print("arr.shape:{0}".format(arr.shape))


# - 2차원 배열(Matrix) 정의

arr2 = np.array([[1,2,3],
                 [4,5,6]]) # 2 행 3열
print(arr2)
print("arr2.shape:{0}".format(arr2.shape))



# - 3차원 배열(Array) 정의
# 2면 2행 3열

arr3 = np.array([[[1,2,3],[4,5,6]],
                 [[7,8,9],[10,11,12]]])
print(arr3)
print("arr3.shape:{0}".format(arr3.shape))

# 배열 생성 및 초기화
# zeros((행,열)) : 0으로 채우는 함수

arr_zeros = np.zeros((3,4))
print(arr_zeros)

# ones((행,열)) : 1로 채우는 함수

arr_ones = np.ones((2,2))
print(arr_ones)

# full((행,열),값) :: 값으로 채우는 함수 // 커스터마이징 형태로 각행이라던지 각열이라던지 따로따로 값을 넣는건 불가능
arr_full = np.full((3,4),7)
print(arr_full)

## zeros 랑 ones 가 있는건 ... 내가봤을땐 2진수 관련됐기 때문에 두개 나눈거고 나머지 full로 채울 수 있게 함수를 정의 한듯 함

# eye(N) : (N,N) 단위 행렬 생성
arr_eye = np.eye(5)
print(arr_eye)

# empty((행,열)) : 초기화 없이 기존 메모리 값이 들어감
arr_empty = np.empty((3,3))
print(arr_empty)


# _like(배열) 지정한 배열과 동일한 shape 의 행렬을 만듦.
# 종류 :: np.zeros_like(), np.ones_like(), np.full_like(), np.empty_like()
arr_sample = np.array([[1,2,3],[4,5,6]])
arr_like = np.ones_like(arr_sample)
print(arr_like)

# 배열 데이터 생성 함수
# - np.linsapce(시작, 종료, 개수) : 개수에 맞게끔 시작과 종료 사이에 균등하게 분배.

arr_linspace = np.linspace(1,10,5)
print(arr_linspace) # [ 1. 3.25 5.5 7.75 10. ]

# plt.plot(arr_linspace, 'o') # 그래프를 그려주는 함수 마커를 원('o')으로 만든 그래프를 보여줌
# plt.show() # 변수 적용 안해도 자동으로 나옴


# np.arange(시작, 종료, 스텝) :: 시작과 종료 사이에 스텝 간격으로 생성
arr_arange = np.arange(1,20,2)
print(arr_arange) # [ 1  3  5  7  9 11 13 15 17 19]

#plt.plot(arr_arange,'v')
#plt.show()

# list vs ndarray(1차원 배열 (Vector))

x1 = [1,2,3]
y1 = [4,5,6]

print(x1 + y1) # [1, 2, 3, 4, 5, 6]

x2 = np.array(x1)
y2 = np.array(y1)

print(x2 + y2) # [5 7 9]

print(type(x1)) #<class 'list'>
print(type(x2)) # <class 'numpy.ndarray'>

print(x2[2]) # 요소의 참조 :: 3 // 메모리쪽으로 접근
x2[2] = 10 # 요소의 수정
print(x2) # [ 1  2 10]

print(np.arange(10)) # range() 와 동일  [0 1 2 3 4 5 6 7 8 9] R -> combine
print(np.arange(5,10)) # [5 6 7 8 9]

x = np.array([10,11,12])

for i in np.arange(1,4):
    print(i)
    print(i+x)

# ndarray 주의
a = np.array([1,1])
b = a # 주소값 복사 배열 -- 참조자료형(자바 생각)

print('a = ' + str(a)) # a = [1 1]
print('b = ' + str(b)) # b = [1 1]

b[0] = 100
print(a) # [100   1]
print(b) # [100   1]

#################################################

a = np.array([1,1])
b = a.copy() # 새로운 메모리 할당 (복사) 후 할당 된 주소 리턴 :: java : clone (인스턴스 복사 개념)

b[0] = 100

print(a) # [1 1]
print(b) # [100   1]

# 행렬(2차원)
x = np.array([[1,2,3],
              [4,5,6]])
print(x)
print(type(x)) # <class 'numpy.ndarray'>
print(x.shape) # (2, 3)

w, h = x.shape
print(w) # 2
print(h) # 3

print(x[1,2]) # 6

x[1,2] = 10

print(x[1,2]) # 10

# 행렬의 크기 변경

a = np.arange(10)
print(a)
a_arange = a.reshape(2,5) # 벡터 -> 매트릭스 변환
print(a_arange) # 10 크기의 백터 형태를 -> 2행 5열 매트릭스 형태로 변환

print(type(a_arange)) #<class 'numpy.ndarray'>

print(a_arange.shape) # (2, 5)

# 행렬(numpy.ndarray)의 사칙연산
#  - 덧셈
x = np.array([[4,4,4],[8,8,8]])
y = np.array([[1,1,1],[2,2,2]])
print(x+y)

'''
[[ 5  5  5]
 [10 10 10]]
'''

# 스칼라 * 행렬
x = np.array([[4,4,4],[8,8,8]])
scar_arr = 10 * x
print(scar_arr)

# - 산술함수 : np.exp() , np.sqrt() , np.log() , np.round() , np.mean(), np.std() , np.max() , np.min()

print(np.exp(x)) # 지수함수 ( y = e ^ x)

# - 행렬 * 행렬 ( [ A * B ] * [ B * x ] = [ A * x ] )
a = np.array([[1,2],[3,4]])
b = np.array([[1,1],[2,2]])
print(a*b)





x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7],[7],[7]])
print(x.dot(y)) # 내적값
'''
[[ 42]
 [105]]
'''

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,2],[7,2],[7,2]])
print(x.dot(y)) # 내적값

'''
[[ 42  12]
 [105  30]]
'''

# 원소 접근
data = np.array([[51,55],[14,19],[0,4]])
print(data)
print(data.shape) # (3, 2)

print(data[0][1]) # 인덱스 접근 방법

for row in data:
    print(row)

'''
[51 55]
[14 19]
[0 4]
'''

y = data.flatten() # Matrix -> vector 평평하게 펴줌
print(y) # [51 55 14 19  0  4]

# 슬라이싱
x = np.arange(10)
print(x[:5]) # [0 1 2 3 4]
print(x[5:]) # [5 6 7 8 9]
print(x[3:8]) # [3 4 5 6 7]
print(x[3:8:2]) # [3 5 7] 2step
print(x[::-1]) # [9 8 7 6 5 4 3 2 1 0] reverse


y = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3행 3열
print(y[:2,1:2]) # [[2] [5]]

# 조건을 만족하는 데이터 수정
# - bool 배열 사용

x = np.array([1,1,2,3,4,5,8,15])

print(x > 3) # [False False False False  True  True  True  True]

y = x[x>3]

print(y) # [ 4  5  8 15]

x[x > 3] = 555

print(x) # [ 1 1 2 3 555 555 555 555]

# -Numpy 에서 np.sum 함수의 axis 이해
arr = np.arange(0, 4 * 2 * 4)
print(len(arr)) # 32

v = arr.reshape([4,2,4]) # 차원변환 (row(x) / axis = 0, col(y) / axis = 1, depth(z) / axis = 2) # 배열측면 :: 4면 2행 4열
print(v)

'''
[[[ 0  1  2  3]
  [ 4  5  6  7]]

 [[ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [20 21 22 23]]

 [[24 25 26 27]
  [28 29 30 31]]]
'''

print(v.shape) # (4, 2, 4)

print(v.ndim) # 차원 수 :: 3
print(v.sum()) # 인자수 합 :: 496

print(v.sum(axis =0)) # row 기준으로 바라볼 것

'''
[[48 52 56 60]
 [64 68 72 76]]
'''

print(v.sum(axis =1)) # colum 기준으로 바라볼 것

'''
[[ 4  6  8 10]
 [20 22 24 26]
 [36 38 40 42]
 [52 54 56 58]]
'''

print(v.sum(axis =2)) # depth 기준으로 바라볼 것

'''
[[  6  22]
 [ 38  54]
 [ 70  86]
 [102 118]]
'''













