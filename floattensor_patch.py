from __future__ import print_function

from floattensor import FloatTensor, LongStorage

import PyCudaTorch
import PyTorchAug
#from PyTorchAug import *

def cuda(self):
#    print('cl')
    res = PyCudaTorch.FloatTensorToCudaTensor(self)
#    print('res', res)
    return res

FloatTensor.cuda = cuda

#PyTorchAug.

#def Linear_cl(self):
#    print('Linear_cl')
#    print('self', self)
#    self.cl()
#    return self

## import PyTorch
#Linear.cl = Linear_cl

print('dir(PyTorchAug)', dir(PyTorchAug))
print('dir(PyCudaTorch)', dir(PyCudaTorch))
PyTorchAug.cythonClasses['torch.CudaTensor'] = {'popFunction': PyCudaTorch.cyPopCudaTensor}
PyTorchAug.populateLuaClassesReverse()

PyTorchAug.pushFunctionByPythonClass[PyCudaTorch.CudaTensor] = PyCudaTorch.cyPushCudaTensor

