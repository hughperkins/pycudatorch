from __future__ import print_function
import PyCudaTorch

#PyCudaTorch.newfunction(123)

import PyTorch
from PyTorchAug import *

def myeval(expr):
    print(expr, ':', eval(expr))

#a = PyTorch.foo(3,2)
#print('a', a)
#print(PyTorch.FloatTensor(3,2))

a = PyTorch.FloatTensor(4, 3).uniform()
print('a', a)
a = a.cuda()
print(type(a))

print('a.dims()', a.dims())
print('a.size()', a.size())
print('a', a)

print('sum:', a.sum())
myeval('a + 1')
b = PyCudaTorch.CudaTensor()
print('got b')
myeval('b')
b.resizeAs(a)
myeval('b')
print('run uniform')
b.uniform()
myeval('b')

print('create new b')
b = PyCudaTorch.CudaTensor()
print('b.dims()', b.dims())
print('b.size()', b.size())
print('b', b)

c = PyTorch.FloatTensor().cuda()
print('c.dims()', c.dims())
print('c.size()', c.size())
print('c', c)

print('creating Linear...')
linear = Linear(3,5)
print('created linear')
print('linear:', linear)
myeval('linear.output')
myeval('linear.output.dims()')
myeval('linear.output.size()')
myeval('linear.output.nElement()')

linear = linear.cuda()
myeval('type(linear)')
myeval('type(linear.output)')
myeval('linear.output.dims()')
myeval('linear.output.size()')
myeval('linear.output')
#print('linearCl.output', linear.output)

output = linear.forward(a)

print('output.dims()', output.dims())
print('output.size()', output.size())

outputFloat = output.float()
print('outputFloat', outputFloat)

print('output', output)

mlp = Sequential()
mlp.add(SpatialConvolutionMM(1,16,5,5,1,1,2,2))
mlp.add(ReLU())
mlp.add(SpatialMaxPooling(3,3,3,3))
mlp.add(SpatialConvolutionMM(16,32,5,5,1,1,2,2))
mlp.add(ReLU())
mlp.add(SpatialMaxPooling(2,2,2,2))
mlp.add(Reshape(32*4*4))
mlp.add(Linear(32*4*4, 150))
mlp.add(Tanh())
mlp.add(Linear(150, 10))
mlp.add(LogSoftMax())

mlp.cuda()

print('mlp', mlp)
myeval('mlp.output')
input = PyTorch.FloatTensor(128,1,28,28).uniform().cuda()
myeval('input[0]')
output = mlp.forward(input)
myeval('output[0]')

