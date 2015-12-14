# pycutorch
POC for Python wrappers for cutorch/cunn

# Example usage

## Pre-requisites

* have cuda in /opt/cuda-7.0 (or modify setup.py to point to different path)
* Have installed torch, per https://github.com/torch/distro
* Have installed cutorch and cunn:
```
luarocks install cutorch
luarocks install cunn
```
* Have python 2.7
* Have setup a virtualenv, with cython, and numpy:
```
virtualenv env
source env/bin/activate
pip install cython
pip install numpy
pip install Jinja2
```
* *You need to have pytorch repo next to pycutorch repository*, ie the parent folder of pycutorch should contain [pytorch](https://github.com/hughperkins/pytorch)

To run:
```
./build.sh
./run.sh
```

Example script:
```
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

```

Output:
```
$ ./run.sh 
/home/user/envs/local/lib/python2.7/site-packages/pkg_resources.py:991: UserWarning: /home/user/.python-eggs is writable by group/others and vulnerable to attack when used with get_resource_filename. Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).
  warnings.warn(msg, UserWarning)
dir(PyTorchAug) ['ClassNLLCriterion', 'Linear', 'LogSoftMax', 'LuaClass', 'MSECriterion', 'PyTorch', 'ReLU', 'Reshape', 'Sequential', 'SpatialConvolutionMM', 'SpatialMaxPooling', 'Tanh', '__builtins__', '__doc__', '__file__', '__loader__', '__name__', '__package__', 'cythonClasses', 'getNextObjectId', 'lua', 'luaClasses', 'luaClassesReverse', 'nextObjectId', 'popString', 'populateLuaClassesReverse', 'print_function', 'pushFunctionByPythonClass', 'pushGlobal', 'pushGlobalFromList', 'pushObject', 'registerObject', 'torchType', 'unregisterObject']
dir(PyCudaTorch) ['CudaGlobalState', 'CudaTensor', 'DoubleTensorToCudaTensor', 'FloatTensorToCudaTensor', 'PyTorch', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'array', 'cyPopCudaTensor', 'cyPushCudaTensor', 'getPrediction', 'numbers']
initializing PyCudaTorch...
loaded cutorch
loaded cunn
 ... PyCudaTorch initialized
a 0.682399 0.991263 0.193337
0.771042 0.859513 0.227952
0.0226214 0.722964 0.338015
0.353466 0.758877 0.573851
[torch.FloatTensor of size 4x3]

<type 'PyCudaTorch.CudaTensor'>
a.dims() 2
a.size()  4
 3
[torch.LongStorage of size 2]

a 0.682399 0.991263 0.193337
0.771042 0.859513 0.227952
0.0226214 0.722964 0.338015
0.353466 0.758877 0.573851
[torch.CudaTensor of size 4x3]

sum: 6.49530076981
a + 1 : 1.6824 1.99126 1.19334
1.77104 1.85951 1.22795
1.02262 1.72296 1.33801
1.35347 1.75888 1.57385
[torch.CudaTensor of size 4x3]

got b
b : [torch.CudaTensor with no dimension]

b : 1.6824 1.99126 1.19334
1.77104 1.85951 1.22795
1.02262 1.72296 1.33801
1.35347 1.75888 1.57385
[torch.CudaTensor of size 4x3]

run uniform
b : 0.245807 0.927532 0.151811
0.720315 0.144195 0.360412
0.122866 0.823565 0.643401
0.374496 0.192979 0.93561
[torch.CudaTensor of size 4x3]

create new b
b.dims() 0
b.size() [torch.LongStorage of size 0]

b [torch.CudaTensor with no dimension]

c.dims() 0
c.size() [torch.LongStorage of size 0]

c [torch.CudaTensor with no dimension]

creating Linear...
Linear.__init__
created linear
linear: nn.Linear(3 -> 5)
linear.output : [torch.DoubleTensor with no dimension]

linear.output.dims() : 0
linear.output.size() : None
linear.output.nElement() : 0
Linear.__init__
type(linear) : <class 'PyTorchAug.Linear'>
type(linear.output) : <type 'PyCudaTorch.CudaTensor'>
linear.output.dims() : 0
linear.output.size() : [torch.LongStorage of size 0]

linear.output : [torch.CudaTensor with no dimension]

output.dims() 2
output.size()  4
 5
[torch.LongStorage of size 2]

outputFloat 0.835138 -0.0984995 0.976471 0.00963038 0.0140738
0.841528 -0.0715615 0.96343 0.0199725 -0.018722
0.665126 -0.0909479 0.698118 -0.12425 0.199515
0.867903 -0.20736 0.928468 -0.0403201 -0.0103005
[torch.FloatTensor of size 4x5]

output 0.835138 -0.0984995 0.976471 0.00963038 0.0140738
0.841528 -0.0715615 0.96343 0.0199725 -0.018722
0.665126 -0.0909479 0.698118 -0.12425 0.199515
0.867903 -0.20736 0.928468 -0.0403201 -0.0103005
[torch.CudaTensor of size 4x5]

Linear.__init__
Linear.__init__
mlp nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): nn.SpatialConvolutionMM(1 -> 16, 5x5, 1,1, 2,2)
  (2): nn.ReLU
  (3): nn.SpatialMaxPooling(3,3,3,3)
  (4): nn.SpatialConvolutionMM(16 -> 32, 5x5, 1,1, 2,2)
  (5): nn.ReLU
  (6): nn.SpatialMaxPooling(2,2,2,2)
  (7): nn.Reshape(512)
  (8): nn.Linear(512 -> 150)
  (9): nn.Tanh
  (10): nn.Linear(150 -> 10)
  (11): nn.LogSoftMax
}
mlp.output : [torch.CudaTensor with no dimension]

input[0][0][0] : 0.951148 0.383791 0.927399 0.896064 0.918963 0.757004 0.527258 0.808004 0.533048 0.880753 0.392538 0.858083 0.204616 0.0887245 0.881087 0.6978 0.717542 0.333986 0.723461 0.970431 0.806659 0.431718 0.32057 0.0817998 0.766709 0.390999 0.0842737 0.936283
[torch.CudaTensor of size 28]

output[0] : -2.41201 -2.48542 -2.16704 -2.25163 -2.26283 -2.21121 -2.34368 -2.32936 -2.41525 -2.19796
[torch.CudaTensor of size 10]

Linear.__init__
Linear.__init__
linear nn.Linear(3 -> 5)
linear.weight 0.126199 0.255241 -0.0918763
-0.34899 0.532427 0.0729166
-0.432081 0.354608 -0.539946
-0.405352 -0.00362546 0.51926
-0.10635 -0.538465 -0.411125
[torch.CudaTensor of size 5x3]

linear.output [torch.CudaTensor with no dimension]

linear.gradInput [torch.CudaTensor with no dimension]

input 0.527763 0.92302 0.252646
0.260251 0.346376 0.156644
0.562014 0.704626 0.220798
0.272335 0.334772 0.0299162
[torch.CudaTensor of size 4x3]

output -0.184994 0.00507969 -0.591282 -0.138991 -0.610176
-0.357116 -0.215582 -0.628341 -0.0783141 -0.231755
-0.233488 -0.125474 -0.666329 -0.16862 -0.483128
-0.34691 -0.235218 -0.569251 -0.148975 -0.17469
[torch.CudaTensor of size 4x5]

gradInput 0.351596 0.0748752 0.515313
0.358055 -0.303671 0.410976
0.421963 -0.101929 0.483152
0.363238 -0.321038 0.316549
[torch.CudaTensor of size 4x3]

criterion nn.ClassNLLCriterion
dir(linear) ['addBuffer', 'bias', 'gradBias', 'gradInput', 'gradWeight', 'output', 'weight']
output -0.184994 0.00507969 -0.591282 -0.138991 -0.610176
-0.357116 -0.215582 -0.628341 -0.0783141 -0.231755
-0.233488 -0.125474 -0.666329 -0.16862 -0.483128
-0.34691 -0.235218 -0.569251 -0.148975 -0.17469
[torch.CudaTensor of size 4x5]

Linear.__init__
Linear.__init__
got criterion
imagesTensor.size()  1280
 784
[torch.LongStorage of size 2]

labelsTensor.size()  1280
[torch.LongStorage of size 1]

type(imagesTensor) <type 'PyCudaTorch.CudaTensor'>
imagesTensor.size()  1280
 1
 28
 28
[torch.LongStorage of size 4]

start training...
epoch 0 accuracy: 64.453125%
epoch 1 accuracy: 91.953125%
epoch 2 accuracy: 96.015625%
epoch 3 accuracy: 97.890625%
epoch 4 accuracy: 99.453125%
epoch 5 accuracy: 99.765625%
epoch 6 accuracy: 99.84375%
epoch 7 accuracy: 99.84375%
epoch 8 accuracy: 99.84375%
epoch 9 accuracy: 99.84375%
epoch 10 accuracy: 99.921875%
epoch 11 accuracy: 100.0%
```

