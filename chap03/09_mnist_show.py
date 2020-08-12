# MNIST(Modified National Institude of Standards and Technology)
# - 손으로 직접 쓴 숫자(필기체 숫자)들로 이루어진 데이터 셋
# - 0 ~ 9까지의 숫자 이미지로 구성되며, 60,000개의 트레이닝 데이터와
#   10,000개의 테스트 데이터로 이루어져 있음.
# - 28x28 size : 픽셀의 사이즈

#이 튜토리얼의 목적은 (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지를 알려주는 것입니다.
#이미지 가져오기
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image #PIL : 직접 네트워크를 통해서 패키지 다운 필요 , pillow 패키지

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #uint8 : unsighed int8(byte) = 음수값 없이 1바이트를 0~255로 제공.
    pil_img.show()

#엔터시 역슬래쉬 자동으로 추가. 다음부터 연속으로 이어진다는 표시. 파이썬은 들여쓰기가 문법으로 적용. 줄바꿈을 했을 때의 여백이 오동작을 할 가능성이 높아서
(x_train, t_train),(x_test, t_test) = \
    load_mnist(flatten=True, normalize=False, one_hot_label=False) #디폴트로 트루 설정, 이름 명시하면 순서는 상관없음. 튜플 형태로 반환.

for i in range(10):
    img = x_train[i] #i번째 인덱스 이미지 읽어오기
    label = t_train[i]
    print(label) #5
    print(img.shape) #(784,)

    img = img.reshape(28,28) # 형상을 원래 이미지의 크기로 변형.
    print(img.shape) #(28,28)

    #img_show(img)