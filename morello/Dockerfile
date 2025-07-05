# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git curl \
      build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev \
      libclang-rt-18-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.88.0
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain "$RUST_VERSION" -y
ENV PATH=/root/.cargo/bin:$PATH

ARG MORELLO_VERSION=main
WORKDIR /
RUN git clone https://github.com/samkaufman/morello.git
WORKDIR /morello
RUN git checkout $MORELLO_VERSION && \
    git submodule update --init --recursive
RUN sed -i 's/^\([[:space:]]*\)const ITERS: u32 = 10;$/\1let ITERS: u32 = std::env::var("CHERRYBENCH_LOOP_STEPS").unwrap().parse::<u32>().unwrap();/' ./morello/examples/matmul_x86.rs && \
  sed -i 's/^\([[:space:]]*\)(result\.best_inner_loop_runtime() \/ result\.inner_loop_iterations)\.as_secs_f64();$/\1result.best_inner_loop_runtime().as_secs_f64();/' ./morello/examples/matmul_x86.rs && \
  grep -q 'result\.best_inner_loop_runtime()\.as_secs_f64();' ./morello/examples/matmul_x86.rs
RUN cargo build --release --no-default-features && \
    cargo build --release --no-default-features --example matmul_x86
ENV CLANG=/usr/bin/clang-18
COPY ./run_bench.sh ./run_matmul_x86_example.sh /