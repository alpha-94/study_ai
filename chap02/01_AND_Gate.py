# AND Gate를 퍼셉트론으로 표현하기

def AND(x1,x2):
    w1, w2, theta =0.5, 0.5, 0.7 # 사람이 직접 매개변수 값을 정함
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    else:
        return 1






if __name__ == '__main__':
    for xs in [(0,0),(0,1),(1,0),(1,1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + ' : ' + str(y))

