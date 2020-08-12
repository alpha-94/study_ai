from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()

mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price , orange_price)
price = mul_tax_layer.forward(all_price, tax)

print('  forward result ↓ \n apple_price :: {0} \n orange_price :: {1} \n all_price :: {2} \n price :: {3} \n'.format(apple_price,orange_price,all_price,int(price)))

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print('  backward result ↓ \
\n dtax :: {0} \
\n dall_price :: {1} \
\n dapple_price :: {2} \
\n dorange_price :: {3} \
\n dapple :: {4} | dapple_num :: {5} \
\n dorange :: {6} | dorange_num :: {7} \
\n'.format(dtax, dall_price,dapple_price,dorange_price,dapple,int(dapple_num),dorange,int(dorange_num)))


