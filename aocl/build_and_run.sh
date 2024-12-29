#!/usr/bin/env bash
set -e 
clang++-18 -std=c++17 -O3 -march=core-avx2 "-DPROBLEM_SIZE=$1" -DNDEBUG \
    -I/opt/aocl/include/blis \
    -L/opt/aocl/lib -lblis-mt -lm -fopenmp \
    -o aocl_bench aocl_bench.cpp
./aocl_bench