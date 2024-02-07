#!/usr/bin/env bash
set -e 
clang++-17 -std=c++17 -O3 -march=core-avx2 "-DPROBLEM_SIZE=$1" -DNDEBUG \
    -I/eigen -o eigen_bench eigen_bench.cpp
./eigen_bench