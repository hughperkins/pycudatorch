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
myeval('input[0][0][0]')
output = mlp.forward(input)
myeval('output[0]')


def test_pycudann():
#    PyTorch.manualSeed(123)
    linear = Linear(3, 5).cuda()
    print('linear', linear)
    print('linear.weight', linear.weight)
    print('linear.output', linear.output)
    print('linear.gradInput', linear.gradInput)

    input = PyTorch.DoubleTensor(4, 3).uniform().cuda()
    print('input', input)
    output = linear.updateOutput(input)
    print('output', output)

    gradInput = linear.updateGradInput(input, output)
    print('gradInput', gradInput)

    criterion = ClassNLLCriterion().cuda()
    print('criterion', criterion)

    print('dir(linear)', dir(linear))

    mlp = Sequential()
    mlp.add(linear)
    mlp.cuda()

    output = mlp.forward(input)
    print('output', output)

    import sys
    sys.path.append('../pytorch/thirdparty/python-mnist')
    from mnist import MNIST
    import numpy
    import array

#    numpy.random.seed(123)

    mlp = Sequential()

    mlp.add(SpatialConvolutionMM(1, 16, 5, 5, 1, 1, 2, 2))
    mlp.add(ReLU())
    mlp.add(SpatialMaxPooling(3, 3, 3, 3))

    mlp.add(SpatialConvolutionMM(16, 32, 3, 3, 1, 1, 1, 1))
    mlp.add(ReLU())
    mlp.add(SpatialMaxPooling(2, 2, 2, 2))

    mlp.add(Reshape(32 * 4 * 4))
    mlp.add(Linear(32 * 4 * 4, 150))
    mlp.add(Tanh())
    mlp.add(Linear(150, 10))
    mlp.add(LogSoftMax())
    mlp.cuda()

    criterion = ClassNLLCriterion().cuda()
    print('got criterion')

    learningRate = 0.02

    mndata = MNIST('/norep/data/mnist')
    imagesList, labelsB = mndata.load_training()
    images = numpy.array(imagesList).astype(numpy.float64)
    #print('imagesArray', images.shape)

    #print(images[0].shape)

    labelsf = array.array('d', labelsB.tolist())
    imagesTensor = PyTorch.asDoubleTensor(images).cuda()

    #imagesTensor = PyTorch.FloatTensor(100,784)
    #labels = numpy.array(20,).astype(numpy.int32)
    #labelsTensor = PyTorch.FloatTensor(100).fill(1)
    #print('labels', labels)
    #print(imagesTensor.size())

    def printStorageAddr(name, tensor):
        print('printStorageAddr START')
        storage = tensor.storage()
        if storage is None:
            print(name, 'storage is None')
        else:
            print(name, 'storage is ', hex(storage.dataAddr()))
        print('printStorageAddr END')

    labelsTensor = PyTorch.asDoubleTensor(labelsf).cuda()
    labelsTensor += 1
    #print('calling size on imagestensor...')
    #print('   (called size)')

    desiredN = 1280
    maxN = int(imagesTensor.size()[0])
    desiredN = min(maxN, desiredN)
    imagesTensor = imagesTensor.narrow(0, 0, desiredN)
    labelsTensor = labelsTensor.narrow(0, 0, desiredN)
    print('imagesTensor.size()', imagesTensor.size())
    print('labelsTensor.size()', labelsTensor.size())
    N = int(imagesTensor.size()[0])
    print('type(imagesTensor)', type(imagesTensor))
    size = PyTorch.LongStorage(4)
    size[0] = N
    size[1] = 1
    size[2] = 28
    size[3] = 28
    imagesTensor.resize(size)
    imagesTensor /= 255.0
    imagesTensor -= 0.2
    print('imagesTensor.size()', imagesTensor.size())

    print('start training...')
    for epoch in range(12):
        numRight = 0
        for n in range(N):
    #        print('n', n)
            input = imagesTensor[n]
            label = labelsTensor[n]
            labelTensor = PyTorch.DoubleTensor(1).cuda()
            labelTensor[0] = label
    #        print('label', label)
            output = mlp.forward(input)
            prediction = PyCudaTorch.getPrediction(output)
    #        print('prediction', prediction)
            if prediction == label:
                numRight += 1
            criterion.forward(output, labelTensor)
            mlp.zeroGradParameters()
            gradOutput = criterion.backward(output, labelTensor)
            mlp.backward(input, gradOutput)
            mlp.updateParameters(learningRate)
    #        PyTorch.collectgarbage()
    #        if n % 100 == 0:
    #            print('n=', n)
        print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')

if __name__ == '__main__':
    test_pycudann()

