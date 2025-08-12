#!/usr/bin/env bash
set -e 

usage() {
    echo "Usage: $0 <benchmark_type> <m> <k> <n>"
    echo "       $0 batch-parallel-f32 <batch_size> <m> <k> <n>"
    echo "       $0 batch-parallel-u8s8s32 <batch_size> <m> <k> <n>"
    echo "  benchmark_type: u8s8s32, f32, batch-parallel-f32, or batch-parallel-u8s8s32"
    echo "  batch_size: number of parallel batches (for batch-parallel-* only)"
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

if [ "$BENCH_TYPE" = "batch-parallel-f32" ] || [ "$BENCH_TYPE" = "batch-parallel-u8s8s32" ]; then
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

BUILD="clang++-18 \
    -std=c++17 \
    -O3 \
    -march=core-avx2 \
    -DNDEBUG \
    -fopenmp \
    -I${MKLROOT:-/opt/intel/oneapi/mkl/latest}/include \
    -L${MKLROOT:-/opt/intel/oneapi/mkl/latest}/lib/intel64 \
    -L/usr/lib/llvm-18/lib \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lomp -lpthread -lm -ldl"

if [ "$BENCH_TYPE" = "f32" ]; then
    $BUILD -o mkl_bench mkl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    $BUILD -o mkl_bench mkl_bench_batch_parallel_f32.cpp
elif [ "$BENCH_TYPE" = "u8s8s32" ]; then
    $BUILD -o mkl_bench mkl_bench_u8s8s32.cpp
elif [ "$BENCH_TYPE" = "batch-parallel-u8s8s32" ]; then
    $BUILD -o mkl_bench mkl_bench_batch_parallel_u8s8s32.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"
    usage
    exit 1
fi

if [ "$BENCH_TYPE" = "batch-parallel-f32" ] || [ "$BENCH_TYPE" = "batch-parallel-u8s8s32" ]; then
    ./mkl_bench "$BATCH_SIZE" "$M" "$K" "$N"
else
    ./mkl_bench "$M" "$K" "$N"
fi
