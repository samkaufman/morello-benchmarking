# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git curl \
      build-essential llvm-17 llvm-17-dev clang-17 lld-17 libomp-17-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.75.0
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain "$RUST_VERSION" -y
ENV PATH=/root/.cargo/bin:$PATH

ARG MORELLO_VERSION=main
WORKDIR /
RUN git clone https://github.com/samkaufman/morello.git \
    && cd morello \
    && git checkout $MORELLO_VERSION \
    && git submodule update --init --recursive
WORKDIR /morello
RUN cargo build --release
ENV CLANG /usr/bin/clang-17
COPY ./run_reformat.sh /
ENTRYPOINT ["/run_reformat.sh"]