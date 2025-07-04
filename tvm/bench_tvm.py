import argparse
import sys
import pathlib
import os
import time
import numpy
import tvm
from tvm import topi
import logging
from tvm import te

logger = logging.getLogger(__name__)

# TODO: Accept llvm args from the command line to ensure these are consistent.
TARGET = "llvm -mcpu=core-avx2"
BN = 32
K_FACTOR = 4
DTYPE = "uint32"


def main():
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("N", type=int)
    args = arg_parser.parse_args()

    device = tvm.device(TARGET, 0)
    os.environ["TVM_NUM_THREADS"] = "1"  # TODO: take as CLI argument

    m, k, n = args.N, args.N, args.N
    func, lowered = make_net(m, k, n)

    out_dir_path = pathlib.Path(os.getenv("CHERRYBENCH_OUTPUT_DIR"))
    with (out_dir_path / "tir.txt").open("w") as fo:
        print(lowered, file=fo)

    outer_steps = int(os.getenv("CHERRYBENCH_OUTER_STEPS", "10"))
    inner_steps = int(os.environ["CHERRYBENCH_LOOP_STEPS"])

    a = tvm.nd.array(numpy.random.rand(1, m, k).astype(DTYPE), device)
    b = tvm.nd.array(numpy.random.rand(1, k, n).astype(DTYPE), device)
    c = tvm.nd.array(numpy.zeros((1, m, n), dtype=DTYPE), device)

    # TODO: Check function correctness.

    func(a, b, c)  # Warm-up
    for _ in range(outer_steps):
        start = time.time()
        for _ in range(inner_steps):
            func(a, b, c)
        duration = time.time() - start
        print(f"{duration:.10f}s")

    # TODO: Make sure the following is semantically equivalent to the above. Compare
    #   to https://github.com/apache/tvm/blob/main/src/runtime/profiling.cc#L862
    #   and https://tvm.apache.org/docs/reference/api/python/runtime.html#tvm.runtime.Module.time_evaluator.
    # TODO: Add a CLI argument for using the TVM evalautor.
    evaluator = func.time_evaluator(func.entry_name, device, number=inner_steps)
    print("TVM time evaluator: %fs" % evaluator(a, b, c).mean, file=sys.stderr)


def make_net(m, k, n):
    A = te.placeholder((1, m, k), dtype=DTYPE, name="A")
    B = te.placeholder((1, k, n), dtype=DTYPE, name="B")

    with tvm.target.Target(TARGET):
        C = topi.nn.batch_matmul(A, B, transpose_a=False, transpose_b=False)

    # Create PrimFunc using the modern TensorIR approach
    func = tvm.build(te.create_prim_func([A, B, C]), target=TARGET)
    lowered = te.create_prim_func([A, B, C])
    return func, lowered


if __name__ == "__main__":
    main()
