import numpy as np

def sigmoid(x):
    return 1.0 /(1+np.exp(-x))

a = np.random.rand(4,4)
b = np.random.rand(4,4)
c=np.hstack((a,b))

print(a,b)
print(sigmoid(a))
print(sigmoid(b))
print(sigmoid(c))

print(np.dot(a,b))
print(a*b)