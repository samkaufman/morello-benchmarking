#!/usr/bin/env bash
target/release/morello bench --inner-loop-iters "$CHERRYBENCH_LOOP_STEPS" "$@" | grep -oP '(?<=loop runtime:).+'