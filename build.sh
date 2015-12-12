#!/bin/bash

# - must have installed torch, and activated
# - must have installed cutorch and cunn:
#    luarocks install cutorch
#    luarocks install cunn
# - must have prerequisites for pytorch installed, ie Cython, numpy, etc
# - pytorch must be cloned to a sibling folder of this folder, ie the parent of pycudatorch should also contain pytorch

# Availalbe env vars:
# NOPYTORCHBUILD=1  => doesnt rebuild pytorch
# NOGIT=1  => doesnt pull from git into pytorch

if [[ x${NOPYTORCHBUILD} == x ]]; then {
    (
      cd ../pytorch
      if [[ x${NOGIT}} == x ]]; then {
          git checkout master
          git pull
      } fi
      pip uninstall -y PyTorch
      ./build.sh || exit 1
      python setup.py install || exit 1
    )
} fi
rm -Rf build cbuild dist *.so *.pyc PyCudaTorch.cpp
#mkdir cbuild
#(cd cbuild; cmake .. && make -j 4 ) || exit 1
pip uninstall -y PyCudaTorch
python setup.py install || exit 1

