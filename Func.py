import numpy as np
import weakref
## Copyright 2021. (이름) All pictures cannot be copied without permission.
## date : 2021 - 07 -21

from Var import Variable

# input : instance of Variable
class Function:
    def __call__(self, *inputs): 
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs :
            output.set_creator(self)

        self.inputs = inputs # save input
        self.outputs =[weakref.ref(output) for output in outputs] # save output

        return outputs if len(outputs) > 1 else outputs[0]

    # i'd not define definition of forward function. 
    # Try to define it through overriding.
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x) :
        return np.array(x)
    return x

# Add function
# number of inputs of Add function is 2 (tuple or list)
#
class Add(Function):
    def forward(self, x0,x1):
        y = x0 +  x1
        return y

    def backward(self, gy):
        return gy, gy


# yeah, i define square function.
# you can do it too.
#
class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

def add(x0,x1):
    return Add()(x0,x1)

def square(x):
    return Square()(x) 

def exp(x):
    return Exp()(x)

## example for one variable function
#
# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(x.data)
# print(y.data)  

# example for multi variable function
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x),square(y))
z.backward()

print(z.data)

print(x.grad)
print(y.grad)      