from Func import Function, add, square
from Var import Variable
import numpy as np

# example 1
x = Variable(np.array(3.0))
y = add(x,x)
y.backward()

print(x.grad) # 2가 나왔으니 성공...

# example 2

x = Variable(np.array(3.0))
y = add(add(x,x),x)

y.backward()

print(x.grad) #3 나왔으니 성공

# example 3

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a),square(a))
y.backward()

print(y.data)
print(x.grad)

#example 4

for i in range(10):
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))

# example 5

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad) # gradient of medium variable is None
print(x0.grad, x1.grad)