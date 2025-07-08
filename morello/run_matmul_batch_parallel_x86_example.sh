#!/usr/bin/env bash
target/release/examples/matmul_batch_parallel_x86 "$@" | grep -oP '(?<=run: )\d+\.\d+(?=s)'
