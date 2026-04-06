# syntax=docker/dockerfile:1
FROM ubuntu:noble AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates clang-18 git libomp-18-dev python3.12 python3.12-dev python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
ENV CC=clang-18 \
    CXX=clang++-18 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /
COPY bench_torch.py /
ENTRYPOINT ["/opt/venv/bin/python3", "/bench_torch.py"]