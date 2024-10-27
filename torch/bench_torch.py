import argparse
import os
import time
import logging
import torch
from torch.linalg import multi_dot

logger = logging.getLogger(__name__)

DTYPE = torch.float32

def main():
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("N", type=int)
    args = arg_parser.parse_args()

    # Use a single CPU thread.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Setting {OMP, MKL}_NUM_THREADS=1 may be unneeded, but won't hurt.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    outer_steps = int(os.getenv("CHERRYBENCH_OUTER_STEPS", "10"))
    inner_steps = int(os.environ["CHERRYBENCH_LOOP_STEPS"])

    a = torch.rand(args.N, args.N, dtype=DTYPE)
    b = torch.rand(args.N, args.N, dtype=DTYPE)
    c = torch.rand(args.N, args.N, dtype=DTYPE)

    multi_dot([a, b, c])  # Warm-up
    for _ in range(outer_steps):
        start = time.time()
        for _ in range(inner_steps):
            multi_dot([a, b, c])
        duration = time.time() - start
        print(f"{duration:.10f}s")

if __name__ == "__main__":
    main()