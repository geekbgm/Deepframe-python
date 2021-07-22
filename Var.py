import numpy as np
from numpy.lib.twodim_base import triu_indices_from
from numpy.lib.utils import deprecate

## Copyright 2021. Gyeonmin Bae All pictures cannot be copied without permission. 
## date  : 2021 - 07 - 21 (Tue)

# input : instance of np.ndarray
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    #for saving creator function
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None
    
    def backward(self, retain_grad = False):
        if self.grad is None : 
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x : x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            
            if not isinstance(gxs,tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad= gx
                else :
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

class Variable2 :
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
                
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    # using while loop
    def backward(self):
        if self.grad is None : 
            self.grad = np.ones_like(self.data)
            
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

x = Variable(np.array(0.5))
x = Variable(None)

