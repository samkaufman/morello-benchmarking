#!/usr/bin/env bash
set -e 

BENCH_TYPE="$1"

if [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    if [ $# -lt 5 ]; then
        echo "Error: batch-parallel-f32 requires 5 arguments: <batch_size> <m> <k> <n>"
        exit 1
    fi
    BATCH_SIZE="$2"
    M="$3"
    K="$4"
    N="$5"
    # Validate batch_size
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Error: batch_size must be a positive integer"
        exit 1
    fi
elif [ "$BENCH_TYPE" = "f32" ]; then
    if [ $# -lt 4 ]; then
        echo "Error: f32 requires 4 arguments: <m> <k> <n>"
        exit 1
    fi
    M="$2"
    K="$3"
    N="$4"
else
    echo "Error: Unknown benchmark type: $BENCH_TYPE"
    echo "Supported types: f32, batch-parallel-f32"
    exit 1
fi

if ! [[ "$M" =~ ^[0-9]+$ ]] || ! [[ "$K" =~ ^[0-9]+$ ]] || ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: M, K, N dimensions must be positive integers"
    exit 1
fi

if [ "$BENCH_TYPE" = "f32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG \
        -I/opt/openblas/include \
        -L/opt/openblas/lib -lopenblas \
        -o openblas_bench openblas_bench_f32.cpp
elif [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG -fopenmp \
        -I/opt/openblas/include \
        -L/opt/openblas/lib -lopenblas \
        -o openblas_bench openblas_bench_batch_parallel_f32.cpp
else
    echo "Unknown benchmark type: $BENCH_TYPE"
    exit 1
fi

if [ "$BENCH_TYPE" = "batch-parallel-f32" ]; then
    ./openblas_bench "$BATCH_SIZE" "$M" "$K" "$N"
else
    ./openblas_bench "$M" "$K" "$N"
fi
