# syntax=docker/dockerfile:1
FROM ubuntu:mantic AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY aocl-linux-aocc-4.1.0_1_amd64.deb /
RUN apt install /aocl-linux-aocc-4.1.0_1_amd64.deb

