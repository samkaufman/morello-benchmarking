#!/usr/bin/env bash
target/release/morello bench "$@" | grep -oP '(?<=Impl Runtime:).+'