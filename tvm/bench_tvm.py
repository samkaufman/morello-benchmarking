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
    # c = tvm.nd.array(numpy.zeros((m, n), dtype=DTYPE), device)

    # TODO: Check function correctness.

    func(a, b)  # Warm-up
    for _ in range(outer_steps):
        start = time.time()
        for _ in range(inner_steps):
            func(a, b)
        duration = time.time() - start
        print(f"{duration:.10f}s")

    # TODO: Make sure the following is semantically equivalent to the above. Compare
    #   to https://github.com/apache/tvm/blob/main/src/runtime/profiling.cc#L862
    #   and https://tvm.apache.org/docs/reference/api/python/runtime.html#tvm.runtime.Module.time_evaluator.
    # TODO: Add a CLI argument for using the TVM evalautor.
    evaluator = func.time_evaluator(func.entry_name, device, number=inner_steps)
    print("TVM time evaluator: %fs" % evaluator(a, b).mean, file=sys.stderr)


def make_net(m, k, n):
    A = te.placeholder((1, m, k), dtype=DTYPE, name="A")
    B = te.placeholder((1, k, n), dtype=DTYPE, name="B")

    with tvm.target.Target(TARGET):
        s = topi.nn.batch_matmul(A, B)

    func = tvm.build(s, [A, B], target=TARGET, name="mmult")
    lowered = tvm.lower(s, [A, B], simple_mode=True)
    return func, lowered

    # -----------
    # TODO: Delete the below unreachable code.
    # -----------

    # Algorithm
    k_reduction = te.reduce_axis((0, k), "k")
    A = te.placeholder((m, k), dtype=DTYPE, name="A")
    B = te.placeholder((k, n), dtype=DTYPE, name="B")
    packedB = te.compute(
        (n / BN, k, BN),
        lambda bigN, k, littleN: B[k, bigN * BN + littleN],
        name="packedB",
    )
    C = te.compute(
        (m, n),
        lambda m, n: te.sum(
            A[m, k] * packedB[n // BN, k, tvm.tir.indexmod(n, BN)], axis=k_reduction
        ),
        name="C",
    )

    s = te.create_schedule(C.op)
    CC = s.cache_write(C, "global")
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], BN, BN)
    s[CC].compute_at(s[C], no)

    mc, nc = s[CC].op.axis

    (kaxis,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(kaxis, factor=K_FACTOR)
    s[CC].reorder(ko, mc, ki, nc)
    s[CC].vectorize(nc)
    s[CC].unroll(ki)

    _, _, littleN = s[packedB].op.axis
    s[packedB].vectorize(littleN)

    func = tvm.build(s, [A, B, C], target=TARGET, name="mmult")
    lowered = tvm.lower(s, [A, B, C], simple_mode=True)
    return func, lowered


if __name__ == "__main__":
    main()
