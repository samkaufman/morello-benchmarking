# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gzip \
      llvm-18 llvm-18-dev clang++-18 lld-18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY program.c /app/
WORKDIR /app
RUN clang-18 -O3 -march=core-avx2 -DNDEBUG -o program program.c
ENTRYPOINT ["./program"]