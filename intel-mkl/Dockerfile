# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates gnupg git curl \
  build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -o /tmp/intel-oneapi.key && \
  gpg --dearmor < /tmp/intel-oneapi.key > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
  rm /tmp/intel-oneapi.key && \
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends intel-oneapi-mkl-devel=2025.2.0 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

FROM base
WORKDIR /app
COPY mkl_bench_f32.cpp mkl_bench_batch_parallel_f32.cpp mkl_bench_u8s8s32.cpp mkl_bench_batch_parallel_u8s8s32.cpp build_and_run.sh ./

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
# Add both MKL and LLVM's lib directories (for libomp.so) to the runtime path.
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/lib/llvm-18/lib

ENTRYPOINT ["./build_and_run.sh"]