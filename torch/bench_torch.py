import argparse
import functools
import logging
import os
import time
from typing import Callable

import torch
from torch.linalg import multi_dot

logger = logging.getLogger(__name__)

DTYPE = torch.float32


def configure_threads(num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


def run_timed_loop(fn: Callable[[], None]) -> None:
    """Warm up once, then time `fn`.

    The given function `fn` must run a loop of CHERRYBENCH_LOOP_STEPS iterations.
    """
    outer_steps = int(os.getenv("CHERRYBENCH_OUTER_STEPS", "10"))
    fn()  # Warm-up
    for _ in range(outer_steps):
        start = time.perf_counter()
        fn()
        duration = time.perf_counter() - start
        print(f"{duration:.10f}s")


def run_compiled_loop(
    loop_fn: Callable[..., torch.Tensor],
    *loop_args: torch.Tensor,
    num_threads: int,
) -> None:
    configure_threads(num_threads)
    inner_steps = int(os.environ["CHERRYBENCH_LOOP_STEPS"])
    iteration_count = torch.tensor(inner_steps, dtype=torch.int64)
    compiled_loop = torch.compile(loop_fn, fullgraph=True)
    run_timed_loop(functools.partial(compiled_loop, *loop_args, iteration_count))


def matmul_chain_compiled_loop(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, iteration_count: torch.Tensor
) -> torch.Tensor:
    """Run the matrix chain inside a compiled while-loop."""

    def cond_fn(i, _):
        return i < iteration_count

    def body_fn(i, _):
        return i + 1, multi_dot([a, b, c])

    _, output = torch.while_loop(
        cond_fn,
        body_fn,
        (
            torch.zeros((), dtype=torch.int64, device=a.device),
            torch.empty_like(a),
        ),
    )
    return output


def softmax_compiled_loop(
    input_tensor: torch.Tensor, iteration_count: torch.Tensor
) -> torch.Tensor:
    """Run repeated softmax inside a compiled while-loop."""

    def cond_fn(i, _):
        return i < iteration_count

    def body_fn(i, _):
        return i + 1, torch.softmax(input_tensor, dim=-1)

    _, output = torch.while_loop(
        cond_fn,
        body_fn,
        (
            torch.zeros((), dtype=torch.int64, device=input_tensor.device),
            torch.empty_like(input_tensor),
        ),
    )
    return output


def main():
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("workload")
    arg_parser.add_argument("args", nargs="*")
    args = arg_parser.parse_args()

    if args.workload == "softmax-f32":
        if len(args.args) != 3:
            raise SystemExit(
                "softmax-f32 requires 3 arguments: <batch_size> <length> <num_threads>"
            )
        batch_size, length, num_threads = (int(arg) for arg in args.args)
        input_tensor = torch.rand(batch_size, length, dtype=DTYPE)
        run_compiled_loop(
            softmax_compiled_loop,
            input_tensor,
            num_threads=num_threads,
        )
    elif args.workload == "matmul2-f32":
        if len(args.args) != 1:
            raise SystemExit("matmul2-f32 requires 1 argument: <size>")
        size = int(args.args[0])
        a = torch.rand(size, size, dtype=DTYPE)
        b = torch.rand(size, size, dtype=DTYPE)
        c = torch.rand(size, size, dtype=DTYPE)
        run_compiled_loop(matmul_chain_compiled_loop, a, b, c, num_threads=1)
    else:
        raise SystemExit(
            f"unknown workload: {args.workload}; expected one of softmax-f32 or matmul2-f32"
        )


if __name__ == "__main__":
    main()
