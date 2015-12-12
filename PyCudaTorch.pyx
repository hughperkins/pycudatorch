from __future__ import print_function

import cython
cimport cython

cimport cpython.array
import array

import PyTorch
cimport PyTorch
cimport Storage

cdef extern from "LuaHelper.h":
    cdef struct lua_State
    void *getGlobal(lua_State *L, const char *name1, const char *name2);
    void require(lua_State *L, const char *name)

cdef extern from "THCGeneral.h":
    cdef struct THCState

cdef extern from "THTensor.h":
    cdef struct THFloatTensor

cdef extern from "THCTensor.h":
    cdef struct THCudaTensor
    THCudaTensor *THCudaTensor_new(THCState *state)
    THCudaTensor *THCudaTensor_newWithSize1d(THCState *state, long size0)
    THCudaTensor *THCudaTensor_newWithSize2d(THCState *state, long size0, long size1)
    THCudaTensor *THCudaTensor_newWithSize3d(THCState *state, long size0, long size1, long size2)
    THCudaTensor *THCudaTensor_newWithSize4d(THCState *state, long size0, long size1, long size2, long size3)
    void THCudaTensor_retain(THCState *state, THCudaTensor*self)
    void THCudaTensor_free(THCState *state, THCudaTensor *tensor)
    int THCudaTensor_nDimension(THCState *state, THCudaTensor *tensor)
    long THCudaTensor_size(THCState *state, const THCudaTensor *self, int dim)
    long THCudaTensor_nElement(THCState *state, const THCudaTensor *self)
    void THCudaTensor_resizeAs(THCState *state, THCudaTensor *self, THCudaTensor *model)
    THCudaTensor *THCudaTensor_newSelect(THCState *state, THCudaTensor *self, int dimension, int sliceIndex)

cdef extern from "THCTensorCopy.h":
    void THCudaTensor_copyFloat(THCState *state, THCudaTensor *self, THFloatTensor *src)
    void THFloatTensor_copyCuda(THCState *state, THFloatTensor *self, THCudaTensor *src)

cdef extern from "THCTensorMath.h":
    float THCudaTensor_sumall(THCState *state, THCudaTensor *self)
    void THCudaTensor_add(THCState *state, THCudaTensor *res, THCudaTensor *self, float scalar)

cdef extern from "cudannWrapper.h":
    THCState *getState(lua_State *L)
    THCudaTensor *popCudaTensor(lua_State *L)
    void pushCudaTensor(THCState *state, lua_State *L, THCudaTensor *tensor)

def cyPopCudaTensor():
    cdef THCudaTensor *tensorC = popCudaTensor(globalState.L)
    cdef CudaTensor tensor = CudaTensor_fromNative(tensorC)
    return tensor

def cyPushCudaTensor(CudaTensor tensor):
    pushCudaTensor(cudaGlobalState.state, globalState.L, tensor.native)

cdef class CudaTensor(object):
    cdef THCudaTensor *native

    def __cinit__(CudaTensor self, *args, _allocate=True):
#        print('CudaTensor.__cinit__')
        if _allocate:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 0:
                self.native = THCudaTensor_new(cudaGlobalState.state)
            elif len(args) == 1:
                self.native = THCudaTensor_newWithSize1d(cudaGlobalState.state, args[0])
            elif len(args) == 2:
                self.native = THCudaTensor_newWithSize2d(cudaGlobalState.state, args[0], args[1])
            elif len(args) == 3:
                self.native = THCudaTensor_newWithSize3d(cudaGlobalState.state, args[0], args[1], args[2])
            elif len(args) == 4:
                self.native = THCudaTensor_newWithSize4d(cudaGlobalState.state, args[0], args[1], args[2], args[3])
            else:
                raise Exception('Not implemented, len(args)=' + str(len(args)))

    def __dealloc__(CudaTensor self):
#        print('CudaTensor.__dealloc__')
        THCudaTensor_free(cudaGlobalState.state, self.native)

    @staticmethod
    def new():
        return CudaTensor()
#        cdef THCudaTensor *newTensorC = THCudaTensor_new(cudaGlobalState.state)
#        return CudaTensor_fromNative(newTensorC, False)

    def __repr__(CudaTensor self):
        cdef PyTorch._FloatTensor floatTensor = self.float()
        floatRepr = floatTensor.__repr__()
        cudaRepr = floatRepr.replace('FloatTensor', 'CudaTensor')
        return cudaRepr

    def float(CudaTensor self):
        cdef PyTorch._FloatTensor floatTensor = PyTorch._FloatTensor.new()
        cdef Storage._LongStorage size = self.size()
        if size is None:
            return PyTorch._FloatTensor()
        if len(size) == 0:
            return PyTorch._FloatTensor()
        floatTensor.resize(size)
        THFloatTensor_copyCuda(cudaGlobalState.state, floatTensor.native, self.native)
        return floatTensor

    def copy(CudaTensor self, PyTorch._FloatTensor src):
        THCudaTensor_copyFloat(cudaGlobalState.state, self.native, src.native)
        return self

    cpdef int dims(CudaTensor self):
        return THCudaTensor_nDimension(cudaGlobalState.state, self.native)

    def size(CudaTensor self):
        cdef int dims = self.dims()
        cdef Storage._LongStorage size
#        print('cltensor.size long versoin')
        if dims >= 0:
            size = Storage._LongStorage(dims)
            for d in range(dims):
                size[d] = THCudaTensor_size(cudaGlobalState.state, self.native, d)
            return size
        else:
            return None  # not sure how to handle this yet

    def nElement(CudaTensor self):
        return THCudaTensor_nElement(cudaGlobalState.state, self.native)

    def sum(CudaTensor self):
        return THCudaTensor_sumall(cudaGlobalState.state, self.native)

    def __add__(CudaTensor self, float scalar):
        cdef CudaTensor res = CudaTensor()
        THCudaTensor_add(cudaGlobalState.state, res.native, self.native, scalar)
        return res

    def __getitem__(CudaTensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef THCudaTensor *res = THCudaTensor_newSelect(cudaGlobalState.state, self.native, 0, index)
        return CudaTensor_fromNative(res, False)

    def resizeAs(CudaTensor self, CudaTensor model):
        THCudaTensor_resizeAs(cudaGlobalState.state, self.native, model.native)
        return self

    def uniform(CudaTensor self, float a=0, float b=1):
        cdef Storage._LongStorage size = self.size()
        cdef PyTorch._FloatTensor floatTensor
#        print('size', size)
        floatTensor = PyTorch._FloatTensor(size)
#        print('got floattensor')
#        print('uniform, floatTensor=', floatTensor)
        floatTensor.uniform(a, b)
        self.copy(floatTensor)
        return self

cdef CudaTensor_fromNative(THCudaTensor *tensorC, retain=True):
    cdef CudaTensor tensor = CudaTensor(_allocate=False )
    tensor.native = tensorC
    if retain:
        THCudaTensor_retain(cudaGlobalState.state, tensorC)
    return tensor

def FloatTensorToCudaTensor(PyTorch._FloatTensor floatTensor):
    cdef Storage._LongStorage size = floatTensor.size()
    cdef CudaTensor clTensor
    cdef int nElement = floatTensor.nElement()
    if nElement > 0:
        if floatTensor.dims() == 1:
            clTensor = CudaTensor(int(size[0]))
        elif floatTensor.dims() == 2:
            clTensor = CudaTensor(int(size[0]), int(size[1]))
        elif floatTensor.dims() == 3:
            clTensor = CudaTensor(int(size[0]), int(size[1]), int(size[2]))
        elif floatTensor.dims() == 4:
            clTensor = CudaTensor(int(size[0]), int(size[1]), int(size[2]), int(size[3]))
        else:
            raise Exception('not implemented')
        clTensor.copy(floatTensor)
        return clTensor
    else:
        return CudaTensor()

import floattensor_patch

cdef PyTorch.GlobalState globalState = PyTorch.getGlobalState()

cdef class CudaGlobalState(object):
    cdef THCState *state

#    def __cinit__(CudaGlobalState self):
#        print('CudaGlobalState.__cinit__')

#    def __dealloc__(self):
#        print('CudaGlobalState.__dealloc__')

cdef CudaGlobalState cudaGlobalState

def init():
    global cudaGlobalState
    cdef THCState *state2
    print('initializing PyCudaTorch...')
    require(globalState.L, 'cutorch')
    print('loaded cutorch')
    require(globalState.L, 'cunn')
    print('loaded cunn')
    cudaGlobalState = CudaGlobalState()
    cudaGlobalState.state = getState(globalState.L)
    print(' ... PyCudaTorch initialized')

init()

