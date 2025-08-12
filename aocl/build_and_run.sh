#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <benchmark_type> <batch_size> <m> <k> <n>"
    echo "  benchmark_type: u8s8s16, f32, batch-parallel-f32 (all take batch_size; use 1 for scalar)"
    echo "  batch_size: number of parallel instances"
    echo "  m: matrix dimension M"
    echo "  k: matrix dimension K"
    echo "  n: matrix dimension N"
}

if [ $# -lt 5 ]; then
    usage
    exit 1
fi

BENCH_TYPE="$1"
BATCH_SIZE="$2";
M="$3"
K="$4"
N="$5"
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: batch_size must be a positive integer"
    usage
    exit 1
fi
if ! [[ "$M" =~ ^[0-9]+$ ]] || ! [[ "$K" =~ ^[0-9]+$ ]] || ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: M, K, N dimensions must be positive integers"
    usage
    exit 1
fi

COMMON_FLAGS="clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG -I/opt/aocl/include/blis -L/opt/aocl/lib -lblis-mt -lm -fopenmp"

if [ "$BENCH_TYPE" = "f32" ]; then
    $COMMON_FLAGS -o aocl_bench aocl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    $COMMON_FLAGS -o aocl_bench aocl_bench_f32.cpp
elif [ "$BENCH_TYPE" = "u8s8s16" ]; then
    $COMMON_FLAGS -o aocl_bench aocl_bench_u8s8s16.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"; usage; exit 1; fi

if [ "$BENCH_TYPE" = "u8s8s16" ]; then
    ./aocl_bench "$M" "$K" "$N"
else
    ./aocl_bench "$BATCH_SIZE" "$M" "$K" "$N"
fi