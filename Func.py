import numpy as np

## Copyright 2021 Signal Processing and Machine Learning at Inha Univ.
## arthor: Gyeongmin Bae
## date : 2021 - 07 -21

from Var import Variable

# input : Variable (reference)
class Function:
    def __call__(self, input):
        x  = input.data
        y = self.forward(x)
        output  = Variable(y)
        return output

    # i'd not define definition of forward function. 
    # Try to define it through overriding.
    def forward(self, x):
        raise NotImplementedError()

# yeah, i define square function through overriding.
# you can do it too.
#
class Square(Function):
    def forward(self, x):
        return x**2

## example
#
x = Variable(np.array(10))
f = Square()
y = f(x)
print(x.data)
print(y)        