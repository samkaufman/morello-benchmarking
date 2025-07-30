# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git curl \
    build-essential llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev \
    gfortran make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM base AS openblas_builder
WORKDIR /build
RUN curl -L https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz | tar xz
WORKDIR /build/OpenBLAS-0.3.30
RUN make -j$(nproc) DYNAMIC_ARCH=1
RUN make install PREFIX=/opt/openblas

FROM base
WORKDIR /app
COPY --from=openblas_builder /opt/openblas /opt/openblas
ENV LD_LIBRARY_PATH=/opt/openblas/lib
ENV PKG_CONFIG_PATH=/opt/openblas/lib/pkgconfig
COPY openblas_bench_f32.cpp openblas_bench_batch_parallel_f32.cpp build_and_run.sh ./
RUN chmod +x build_and_run.sh
ENTRYPOINT ["/app/build_and_run.sh"]
