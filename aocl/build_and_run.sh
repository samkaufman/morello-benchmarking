#!/usr/bin/env bash
set -e 

usage() {
    echo "Usage: $0 <benchmark_type> <m> <k> <n>"
    echo "  benchmark_type: u8s8s16 or f32"
    echo "  m: matrix dimension M"
    echo "  k: matrix dimension K"
    echo "  n: matrix dimension N"
}

# Simple argument parsing: benchmark_type m k n
if [ $# -lt 4 ]; then
    usage
    exit 1
fi

BENCH_TYPE="$1"
M="$2"
K="$3"
N="$4"

# Validate M, K, N
if ! [[ "$M" =~ ^[0-9]+$ ]] || ! [[ "$K" =~ ^[0-9]+$ ]] || ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: M, K, N dimensions must be positive integers"
    usage
    exit 1
fi

if [ "$BENCH_TYPE" = "f32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG \
        -I/opt/aocl/include/blis \
        -L/opt/aocl/lib -lblis-mt -lm -fopenmp \
        -o aocl_bench aocl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "u8s8s16" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG \
        -I/opt/aocl/include/blis \
        -L/opt/aocl/lib -lblis-mt -lm -fopenmp \
        -o aocl_bench aocl_bench_u8s8s16.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"
    usage
    exit 1
fi

./aocl_bench "$M" "$K" "$N"