import numpy as np
from Func import Function 

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

