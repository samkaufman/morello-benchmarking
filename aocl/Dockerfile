# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git curl \
      build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install AOCL
FROM base AS build-aocl
ARG AOCL_VERSION
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# TODO: put in /usr/src?
RUN git clone --branch ${AOCL_VERSION} --depth 1 https://github.com/amd/blis.git /blis-src
WORKDIR /blis-src
RUN ./configure --prefix=/opt/aocl \
    -a aocl_gemm --enable-cblas --enable-threading=openmp \
    --enable-shared --disable-static \
    CC=gcc CXX=g++ amdzen \
  && make -j \
  && make check
RUN make install

FROM base
COPY --from=build-aocl /opt/aocl /opt/aocl
WORKDIR /app
COPY aocl_bench_u8s8s16.cpp aocl_bench_f32.cpp build_and_run.sh ./
RUN chmod u=rx,g=,o= build_and_run.sh
ENV BLIS_ARCH_DEBUG=1 BLIS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/aocl/lib
ENTRYPOINT ["./build_and_run.sh"]