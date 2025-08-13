#!/usr/bin/env bash
set -e

usage() {
    cat <<EOF
Usage:
    $0 u8s8s16 <m> <k> <n>
    $0 batch-parallel-f32 <batch_size> <m> <k> <n>

Args:
    benchmark_type: u8s8s16 | batch-parallel-f32
    batch_size: number of instances to run in parallel
    m,k,n: positive integer matrix dimensions
EOF
}

if [ $# -lt 2 ]; then
    usage; exit 1; fi

BENCH_TYPE="$1"; shift

case "$BENCH_TYPE" in
    u8s8s16)
        [ $# -eq 3 ] || { echo "u8s8s16 requires: m k n"; usage; exit 1; }
        BATCH_SIZE=1
        M="$1"; K="$2"; N="$3" ;;
    batch-parallel-f32)
        [ $# -eq 4 ] || { echo "batch-parallel-f32 requires: <batch_size> <m> <k> <n>"; usage; exit 1; }
        BATCH_SIZE="$1"; M="$2"; K="$3"; N="$4" ;;
    *)
        echo "Unknown benchmark type: $BENCH_TYPE"; usage; exit 1 ;;
esac

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "Error: batch_size must be a positive integer"; exit 1; fi
if ! [[ "$M" =~ ^[0-9]+$ ]] || ! [[ "$K" =~ ^[0-9]+$ ]] || ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: M, K, N must be positive integers"; exit 1; fi

COMMON_FLAGS="clang++-18 -std=c++17 -O3 -march=core-avx2 -DNDEBUG -I/opt/aocl/include/blis -L/opt/aocl/lib -lblis-mt -lm -fopenmp"

case "$BENCH_TYPE" in
    batch-parallel-f32) $COMMON_FLAGS -o aocl_bench aocl_bench_f32.cpp ;;
    u8s8s16) $COMMON_FLAGS -o aocl_bench aocl_bench_u8s8s16.cpp ;;
esac

if [ "$BENCH_TYPE" = "u8s8s16" ]; then
    ./aocl_bench "$M" "$K" "$N"
else
    ./aocl_bench "$BATCH_SIZE" "$M" "$K" "$N"
fi