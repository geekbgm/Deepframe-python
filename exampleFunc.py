from Var import Variable, Variable2
import numpy as np
from Func import Function, Square 

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

def numerical_diff(f,x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(x1.data - x0.data)

# comfortable function
def square(x) : 
    f = Square()
    return f(x)

def exp(x) : 
    f= Exp()
    return f(x)


## example of forward computational graph
#
A=Square()

B=Exp()

C=Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)

print(c.data)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f,x)
print(dy)

## example of numerical_diff
#
def f(x) : 
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f,x)
print(dy)

## example of numerical_diff with chain rule
#
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad) # b.grad = C.grad * y.grad
a.grad = B.backward(b.grad) # a.grad = B.grad * b.grad
x.grad = A.backward(a.grad) # x.grad = A.grad * a.grad
print(x.grad) # right answer

# x.grad = A.grad * B.grad * C.grad * y.grad 

## example of auto numerical_diff with chain rule and creator

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a = A(x)
b = B(a)
y = C(b)

#check
assert y.creator == C #True
assert y.creator.input == b #True
assert y.creator.input.creator == B #True  
assert y.creator.input.creator.input == a #True
assert y.creator.input.creator.input.creator == A #True
assert y.creator.input.creator.input.creator.input == x #True

# starting computing gradient using creator
y.grad = np.array(1.0) # gradient of output = dy/dy =1

C = y.creator # y's creator is C 
b = C.input # C's input is b
b.grad = C.backward(y.grad) # b's gradient is C's backward() ; b.grad = C.grad * y.grad

B = b.creator
a = B.input
a.grad = B.backward(b.grad) # a.grad = B.grad * b.grad

A = a.creator
x = A.input
x.grad = A.backward(a.grad) # x.grad = A.grad * a.grad

print(x.grad)

# x.grad = A.grad * B.grad * C.grad * y.grad

# why do we create concept about creator?
# follow me, i will show you why

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# automatic backpropagation with recursion
y.grad = np.array(1.0)
y.backward()
print(x.grad)

# automatic backpropagation with loop
x = Variable2(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

# automatic backpropagation with comfortable function
x = Variable2(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

# automatic backpropagation 2 with comfortable function
x = Variable2(np.array(0.5))
y = square(exp(square(x)))

y.grad = np.array(1.0)
y.backward()
print(x.grad)



