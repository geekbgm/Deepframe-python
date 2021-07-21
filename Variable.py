import numpy as np

## Copy right 2021 Signal Processing and Machine learning at Inha Unv.
## arthor : Gyeongmin Bae
## date  : 2021 - 07 - 21 (Tue)

# input : multi-dimensional array(tensor)
class Variable:
    def __init__(self, data):
        self.data = data


# here is example, 
#
data = np.array(1.0) # made 0-dimensional data
x = Variable(data) # push to variable 
print(x.data) # print it

# Variable is able to overriding. 
# This is good point for ours. 
# right? try it yourself with your own code please 
x.data = np.array(2.0)
print(x.data)