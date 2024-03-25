# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gzip \
      llvm-17 llvm-17-dev clang++-17 lld-17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake make git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY gemma_tiny ./

RUN cmake -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF \
    -DWEIGHT_TYPE=hwy::bfloat16_t \
    -DCMAKE_C_COMPILER=clang-17 -DCMAKE_C_FLAGS="-O3 -march=core-avx2" \
    -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_CXX_FLAGS="-O3 -march=core-avx2" \
    -DCMAKE_BUILD_TYPE=Release -B build
RUN make -j -C build

ENTRYPOINT ["./build/run_bench"]