# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gzip \
      llvm-17 llvm-17-dev clang++-17 lld-17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY program.c /app/
WORKDIR /app
RUN clang-17 -O3 -march=core-avx2 -DNDEBUG -o program program.c
ENTRYPOINT ["./program"]