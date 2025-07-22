#!/usr/bin/env bash
target/release/examples/matmul_x86_parameterized "$@" | grep -oP '(?<=run: )\d+\.\d+(?=s)'