#!/usr/bin/env bash
set -e 

usage() {
    echo "Usage: $0 <benchmark_type> <m> <k> <n>"
    echo "       $0 batch-parallel-f32 <batch_size> <m> <k> <n>"
    echo "  benchmark_type: u8s8s32, f32, or batch-parallel-f32"
    echo "  batch_size: number of parallel batches (for batch-parallel-f32 only)"
    echo "  m: matrix dimension M"
    echo "  k: matrix dimension K"
    echo "  n: matrix dimension N"
}

# Simple argument parsing: benchmark_type [batch_size] m k n
if [ $# -lt 4 ]; then
    usage
    exit 1
fi

BENCH_TYPE="$1"

if [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    if [ $# -lt 5 ]; then
        usage
        exit 1
    fi
    BATCH_SIZE="$2"
    M="$3"
    K="$4"
    N="$5"
    # Validate batch_size
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Error: batch_size must be a positive integer"
        usage
        exit 1
    fi
else
    M="$2"
    K="$3"
    N="$4"
fi

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
elif [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG -fopenmp \
        -I/usr/include/mkl \
        -L/usr/lib/x86_64-linux-gnu/mkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl \
        -o mkl_bench mkl_bench_batch_parallel_f32.cpp
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

if [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    ./mkl_bench "$BATCH_SIZE" "$M" "$K" "$N"
else
    ./mkl_bench "$M" "$K" "$N"
fi
