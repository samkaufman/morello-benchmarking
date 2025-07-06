#!/usr/bin/env bash
set -e 

usage() {
    echo "Usage: $0 <benchmark_type> <problem_size>"
    echo "  benchmark_type: u8s8s32 or f32"
    echo "  problem_size: positive integer"
}

# Simple argument parsing: benchmark_type problem_size
if [ $# -lt 2 ]; then
    usage
    exit 1
fi

BENCH_TYPE="$1"
PROBLEM_SIZE="$2"

# Validate problem size
if ! [[ "$PROBLEM_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: Problem size '$PROBLEM_SIZE' must be a positive integer"
    usage
    exit 1
fi

if [ "$BENCH_TYPE" = "f32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 "-DPROBLEM_SIZE=$PROBLEM_SIZE" -DNDEBUG \
        -I/usr/include/mkl \
        -L/usr/lib/x86_64-linux-gnu/mkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl \
        -o mkl_bench mkl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "u8s8s32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 "-DPROBLEM_SIZE=$PROBLEM_SIZE" -DNDEBUG \
        -I/usr/include/mkl \
        -L/usr/lib/x86_64-linux-gnu/mkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl \
        -o mkl_bench mkl_bench_u8s8s32.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"
    usage
    exit 1
fi

./mkl_bench
