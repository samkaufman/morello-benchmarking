# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      git python3.12 python3.12-dev python3-setuptools libtinfo-dev \
      python3-numpy python3-decorator python3-attr python3-scipy \
      python3-psutil python3-typing-extensions \
      zlib1g-dev build-essential cmake libedit-dev libxml2-dev ninja-build \
      llvm-17 llvm-17-dev clang-17 lld-17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download TVM v0.14.0 (7315c9d)
RUN git clone --recursive https://github.com/apache/tvm.git /tvm \
    && cd /tvm \
    git checkout 7315c9d
WORKDIR /tvm

# Build TVM
RUN update-alternatives    --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100 \
                           --slave   /usr/bin/clang   clang   /usr/bin/clang-17       \
    && update-alternatives --install /usr/bin/cc      cc      /usr/bin/clang      100 \
    && update-alternatives --install /usr/bin/c++     c++     /usr/bin/clang++    100 \
    && update-alternatives --install /usr/bin/ld      ld      /usr/bin/lld-17     100
RUN mkdir build \
    && cd build \
    && cmake .. -G Ninja -DUSE_LLVM=ON CC=clang-17 CXX=clang++-17 \
    && ninja

WORKDIR /
COPY bench_tvm.py /
ENV PYTHONPATH=/tvm/python
ENTRYPOINT ["python3", "/bench_tvm.py"]