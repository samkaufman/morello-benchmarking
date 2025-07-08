#!/usr/bin/env bash
target/release/examples/matmul_x86 "$@" | grep -oP '(?<=run: )\d+\.\d+(?=s)'