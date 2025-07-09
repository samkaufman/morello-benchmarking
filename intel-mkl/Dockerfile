# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates git curl \
  build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev \
  intel-mkl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

FROM base
WORKDIR /app
COPY mkl_bench_f32.cpp mkl_bench_batch_parallel_f32.cpp mkl_bench_u8s8s32.cpp mkl_bench_batch_parallel_u8s8s32.cpp build_and_run.sh ./
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mkl MKL_NUM_THREADS=1
ENTRYPOINT ["./build_and_run.sh"]