# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      git python3.12 python3.12-dev python3-setuptools python3-pip libtinfo-dev \
      python3-numpy python3-decorator python3-attr python3-scipy \
      python3-psutil python3-typing-extensions python3-cloudpickle cython3 \
      zlib1g-dev build-essential cmake libedit-dev libxml2-dev ninja-build \
      llvm-18 llvm-18-dev clang-18 lld-18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download TVM
RUN git clone --recursive --branch v0.20.0 --depth 1 https://github.com/apache/tvm.git /tvm
WORKDIR /tvm

# Build TVM
RUN update-alternatives    --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
                           --slave   /usr/bin/clang   clang   /usr/bin/clang-18       \
    && update-alternatives --install /usr/bin/cc      cc      /usr/bin/clang      100 \
    && update-alternatives --install /usr/bin/c++     c++     /usr/bin/clang++    100 \
    && update-alternatives --install /usr/bin/ld      ld      /usr/bin/lld-18     100
RUN mkdir build \
    && cd build \
    && cmake .. -G Ninja -DUSE_LLVM=ON CC=clang-18 CXX=clang++-18 \
    && ninja

# MetaSchedule tuning dependency; xgboost is not packaged by apt.
RUN python3 -m pip install --break-system-packages --no-cache-dir "xgboost<2"

WORKDIR /
COPY bench_tvm.py /
ENV PYTHONPATH=/tvm/python
ENTRYPOINT ["python3", "/bench_tvm.py"]