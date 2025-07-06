#!/usr/bin/env bash
set -e 

usage() {
    echo "Usage: $0 <benchmark_type> <m> <k> <n>"
    echo "  benchmark_type: u8s8s32 or f32"
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
        -I/usr/include/mkl \
        -L/usr/lib/x86_64-linux-gnu/mkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl \
        -o mkl_bench mkl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "u8s8s32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG \
        -I/usr/include/mkl \
        -L/usr/lib/x86_64-linux-gnu/mkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl \
        -o mkl_bench mkl_bench_u8s8s32.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"
    usage
    exit 1
fi

./mkl_bench "$M" "$K" "$N"
