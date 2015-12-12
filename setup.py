# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
import sys
import platform
from setuptools import setup
from setuptools import Extension

home_dir = os.getenv('HOME')
print('home_dir:', home_dir)

cython_present = False
from Cython.Build import cythonize
cython_present = True

building_dist = False
for arg in sys.argv:
    if arg in ('sdist', 'bdist', 'bdist_egg', 'build_ext'):
        building_dist = True
        break

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
    compile_options.append('/EHsc')
elif osfamily == 'Linux':
    compile_options.append('-std=c++0x')
    compile_options.append('-g')
    if 'DEBUG' in os.environ:
        compile_options.append('-O0')
else:
    pass
    # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
#libraries.append('mylib')
#libraries.append('clnnWrapper')
libraries.append('TH')
libraries.append('THC')
libraries.append('TorchLanguageIndependence')
#libraries.append('PyTorch')
library_dirs = []
library_dirs.append('cbuild')
library_dirs.append(home_dir + '/torch/install/lib')

if osfamily == 'Linux':
    runtime_library_dirs = ['.']

if osfamily == 'Windows':
    libraries.append('winmm')

sources = ["PyCudaTorch.cxx", "cudannWrapper.cpp"]
if cython_present:
    sources = ["PyCudaTorch.pyx", "cudannWrapper.cpp"]
ext_modules = [
    Extension("PyCudaTorch",
              sources=sources,
              include_dirs=[
                  '/opt/cuda-7.0/include',
                  home_dir + '/torch/install/include',
                  home_dir + '/torch/install/include/TH',
                  home_dir + '/torch/install/include/THC',
                  '/usr/include/lua5.1',
                  '../pytorch/src'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"),
#    Extension("GlobalState",
#              sources=['GlobalState.pyx'],
#              include_dirs=[
#                  home_dir + '/torch/install/include/TH',
#                  home_dir + '/torch/install/include/THCl',
#                  '../pytorch'],
#              library_dirs=library_dirs,
#              libraries=libraries,
#              extra_compile_args=compile_options,
#              runtime_library_dirs=runtime_library_dirs,
#              language="c++"),
]

ext_modules = cythonize(ext_modules)

setup(
    name='PyCudaTorch',
    version='',
    author="Hugh Perkins",
    author_email="hughperkins@gmail.com",
    description=(
        'Python wrappers for cutorch'),
    license='BSD2',
    url='https://github.com/hughperkins/pycudatorch',
    long_description='',
    classifiers=[
    ],
    install_requires=[],
    scripts=[],
    ext_modules=ext_modules,
)

