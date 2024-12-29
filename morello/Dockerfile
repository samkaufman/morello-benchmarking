# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git curl \
      build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.83.0
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain "$RUST_VERSION" -y
ENV PATH=/root/.cargo/bin:$PATH

ARG MORELLO_VERSION=main
WORKDIR /
RUN git clone https://github.com/samkaufman/morello.git \
    && cd morello \
    && git checkout $MORELLO_VERSION \
    && git submodule update --init --recursive
WORKDIR /morello
RUN cargo build --release -F verification
ENV CLANG=/usr/bin/clang-18
COPY ./run_reformat.sh /
ENTRYPOINT ["/run_reformat.sh"]