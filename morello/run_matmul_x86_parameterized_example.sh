#!/usr/bin/env bash
binary_path=$(target/release/examples/matmul_x86_parameterized "$@" | tail -n1)
if [ ! -f "$binary_path" ]; then
    echo "Error: Binary not found at $binary_path" >&2
    exit 1
fi

# perf record -e task-clock,cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,l1_dtlb_misses,l2_dtlb_misses,cache-misses,cache-references,l2_fill_pending.l2_fill_busy,l2_latency.l2_cycles_waiting_on_fills -o "$CHERRYBENCH_OUTPUT_DIR/perf.data" "$binary_path" > /dev/null
perf stat -e task-clock,cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,l1_dtlb_misses,l2_dtlb_misses,cache-misses,cache-references,l2_fill_pending.l2_fill_busy,l2_latency.l2_cycles_waiting_on_fills --quiet -o "$CHERRYBENCH_OUTPUT_DIR/perf.data" record "$binary_path" "$CHERRYBENCH_LOOP_STEPS" > /dev/null
for _ in {1..10}; do
    "$binary_path" "$CHERRYBENCH_LOOP_STEPS" | grep -oP 'cpu: \d+s \d+ns' | awk '{gsub(/[^0-9]/, " ", $2); gsub(/[^0-9]/, " ", $3); print $2 + $3/1000000000}'
done