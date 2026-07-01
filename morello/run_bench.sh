#!/usr/bin/env bash
export RAYON_NUM_THREADS=$1
TARGET=$2
DB=$3
shift 3

target/release/morello --target "$TARGET" --db "$DB" --cache-size 50000 bench --inner-loop-iters "$CHERRYBENCH_LOOP_STEPS" "$@" | grep -oP '(?<=loop runtime:).+'