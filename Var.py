import numpy as np
from numpy.lib.twodim_base import triu_indices_from
from numpy.lib.utils import deprecate

## Copyright 2021 Signal Processing and Machine learning at Inha Unv.
## arthor : Gyeongmin Bae
## date  : 2021 - 07 - 21 (Tue)

# input of variable : multi-dimensional array(tensor)
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    #for saving creator function
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Variable2 :
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        func = self.creator
        while func : 
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)

            if x.creator is not None : 
                func = x.creator

# here is example, 
#
data = np.array(1.0) # made 0-dimensional data
x = Variable(data) # push to variable 
# print(x.data) # print it

# Variable is able to overriding. 
# This is good point for ours. 
# right? try it yourself with your own code please 
x.data = np.array(2.0)
# print(x.data)