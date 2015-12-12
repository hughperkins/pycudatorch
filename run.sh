#!/bin/bash

if [[ x$RUNGDB == x ]]; then {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild python test_cudatorch.py
} else {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild rungdb.sh python test_cudatorch.py
} fi

