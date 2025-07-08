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
COPY ./matmul_x86_benchmarking.patch ./
RUN git apply matmul_x86_benchmarking.patch
RUN sed -i 's/^fn apply_to_leaf_spec/pub fn apply_to_leaf_spec/' morello/src/scheduling_sugar.rs
RUN sed -i -E 's/\.split\(([0-9]+)\)/.split_saturating(\1)/g' morello/examples/matmul_x86.rs
RUN sed -i -E 's/\.tile_out\(([^)]+)\)/.tile_out_saturating(\1)/g' morello/examples/matmul_x86.rs
COPY ./new_epi_matmul_x86.rs /
RUN cat /new_epi_matmul_x86.rs >> morello/examples/matmul_x86.rs
RUN cargo build --release --no-default-features && \
    cargo build --release --no-default-features --example matmul_x86
ENV CLANG=/usr/bin/clang-18
COPY ./run_bench.sh ./run_matmul_x86_example.sh /