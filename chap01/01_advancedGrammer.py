# 데이터 분석에 유용한 기능들

# - comprhension 기본 구조

numbers = [1,2,3,4,5]

## 방법 1 - 1
square1 = []
for i in numbers:
    square1.append(i**2)
print(square1)

## 방법 1 - 2
square2 = [i**2 for i in numbers]
print(square2)

## 방법 2 - 1
square3 = []
for i in numbers:
    if i >= 3:
        square3.append(i**2)
print(square3)

## 방법 2 - 2
square4 = [i**2 for i in numbers if i>= 3]
print(square4)

# - split(구분자) :: 구분자로 구분, 기본값은 공백
test_text = 'the-joeun-computer-with-python'
result = test_text.split('-')
print(result) # ['the', 'joeun', 'computer', 'with', 'python'] => list 형태

# 구분자.join(리스트) :: split 함수와 반대로 구분자로 붙인다.
test_text = ['the', 'joeun', 'computer', 'with', 'python']
print(test_text)

result = '-'.join(test_text)
print(result) # the-joeun-computer-with-python

# split() 와 join() 의 응용
result = '-'.join('345.234.6789'.split('.'))
print(result) # 345-234-6789

# enumerate(list) :: 인덱스와 값을 함께 반환
for i, name in enumerate(['a','b','c','d']): # unpacking
    print(i,name)

seq = ['mon','tue','wed','thu','fri','sat','sun']
print(dict(enumerate(seq))) # index -> key

key_seq = 'abcdefg'
value_seq = ['mon','tue','wed','thu','fri','sat','sun']

dict = dict(zip(key_seq,value_seq))
print(dict) # {'a': 'mon', 'b': 'tue', 'c': 'wed', 'd': 'thu', 'e': 'fri', 'f': 'sat', 'g': 'sun'}

day = ['mon','tue','wed','thu','fri','sat','sun']
print([x for x in day])

data = [35, 56, -53, 45, 27, -28, 8, -12]
print([i for i in data if i>=0])
print([i**2 for i in data if i>=0])

# Count 를 이용한 카운팅
#   - Countsms 아이템의 갯수를 자동으로 카운팅 .
from collections import Counter

message = '''
대법원장은 국회의 동의를 얻어 대통령이 임명한다. 언론·출판에 대한 허가나 검열과 집회·결사에 대한 허가는 인정되지 아니한다.
각급 선거관리위원회는 선거인명부의 작성등 선거사무와 국민투표사무에 관하여 관계 행정기관에 필요한 지시를 할 수 있다. 모든 국민은 근로의 권리를 가진다.
국가는 사회적·경제적 방법으로 근로자의 고용의 증진과 적정임금의 보장에 노력하여야 하며, 법률이 정하는 바에 의하여 최저임금제를 시행하여야 한다.
환경권의 내용과 행사에 관하여는 법률로 정한다. 국방상 또는 국민경제상 긴절한 필요로 인하여 법률이 정하는 경우를 제외하고는, 사영기업을 국유 또는 공유로 이전하거나
그 경영을 통제 또는 관리할 수 없다.
'''

counter = Counter(message.split()) # default :: 공백
print(counter) # 자체적으로 빈도를 찾아줌 {'또는': 3, '대한': 2, '수': 2,  ... } -> 딕셔너리 형태
print(type(counter)) # <class 'collections.Counter'>

# Counter (dict) -> list 형태로 반환
print(counter.most_common()) # [('또는', 3), ('대한', 2), ('수', 2), ('법률이', 2), ('정하는', 2),...] -> 리스트 형태

# list -> dict 형태로 반환
#error :: TypeError: 'dict' object is not callable counter.most_common() dict 형변환 지원 불가


'''
dict_msg = dict(counter.most_common())
print(dict_msg)
'''





















