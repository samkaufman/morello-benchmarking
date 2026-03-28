# syntax=docker/dockerfile:1
FROM ubuntu:noble AS builder

ARG XNNPACK_COMMIT=77596e7d8bd21bed7748e4369a701c40be568fce
ENV XNNPACK_DIR=/src/XNNPACK \
    XNNPACK_BUILD_DIR=/src/XNNPACK/build

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl \
    build-essential cmake ninja-build python3 \
    llvm-18 llvm-18-dev clang-18 lld-18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN mkdir -p "$XNNPACK_DIR" \
    && curl -L "https://github.com/google/XNNPACK/archive/${XNNPACK_COMMIT}.tar.gz" \
    | tar xz --strip-components=1 -C "$XNNPACK_DIR"

RUN cmake -G Ninja -S "$XNNPACK_DIR" -B "$XNNPACK_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-18 \
    -DCMAKE_CXX_COMPILER=clang++-18 \
    -DXNNPACK_LIBRARY_TYPE=static \
    -DXNNPACK_BUILD_ALL_MICROKERNELS=OFF \
    -DXNNPACK_BUILD_BENCHMARKS=OFF \
    -DXNNPACK_BUILD_TESTS=OFF \
    && cmake --build "$XNNPACK_BUILD_DIR" --parallel --target XNNPACK

WORKDIR /src/benchmark
COPY xnnpack_softmax_bench_f32.cpp ./
RUN clang++-18 -std=c++17 -O3 -DNDEBUG \
    -I/src/XNNPACK/include \
    -I/src/XNNPACK/build/pthreadpool-source/include \
    -o xnnpack_softmax_bench \
    xnnpack_softmax_bench_f32.cpp \
    /src/XNNPACK/build/libXNNPACK.a \
    /src/XNNPACK/build/libxnnpack-microkernels-prod.a \
    /src/XNNPACK/build/pthreadpool/libpthreadpool.a \
    /src/XNNPACK/build/cpuinfo/libcpuinfo.a \
    -lm -lpthread

FROM ubuntu:noble

WORKDIR /app
COPY --from=builder /src/benchmark/xnnpack_softmax_bench /app/
COPY build_and_run.sh /app/
RUN chmod +x /app/build_and_run.sh

ENTRYPOINT ["/app/build_and_run.sh"]
