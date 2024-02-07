# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gzip \
      llvm-17 llvm-17-dev clang++-17 lld-17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /eigen.tar.bz2 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    && mkdir /eigen \
    && tar -xf /eigen.tar.bz2 -C /eigen --strip-components=1 \
    && rm /eigen.tar.bz2 \
    && mkdir /app

WORKDIR /app
COPY eigen_bench.cpp build_and_run.sh ./
RUN chmod u=rx,g=,o= build_and_run.sh
ENTRYPOINT ["./build_and_run.sh"]