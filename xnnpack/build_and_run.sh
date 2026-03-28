#!/usr/bin/env bash
set -e

BENCH_TYPE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$BENCH_TYPE" != "f32" ]; then
    echo "Error: unknown benchmark type: $BENCH_TYPE"
    exit 1
fi

if [ $# -ne 4 ]; then
    echo "Error: $BENCH_TYPE requires 4 arguments: <batch_size> <channels> <num_cores>"
    exit 1
fi

BATCH_SIZE="$2"
CHANNELS="$3"
NUM_CORES="$4"

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "Error: batch_size must be a positive integer"
    exit 1
fi

if ! [[ "$CHANNELS" =~ ^[0-9]+$ ]] || [ "$CHANNELS" -lt 1 ]; then
    echo "Error: channels must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]] || [ "$NUM_CORES" -lt 1 ]; then
    echo "Error: num_cores must be a positive integer"
    exit 1
fi

if [ "$NUM_CORES" -gt "$BATCH_SIZE" ]; then
    echo "Error: num_cores must be less than or equal to batch_size"
    exit 1
fi

exec "$SCRIPT_DIR/xnnpack_softmax_bench" "$BATCH_SIZE" "$CHANNELS" "$NUM_CORES"
