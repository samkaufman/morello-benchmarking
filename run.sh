#!/usr/bin/env bash
set -e

poetry install --sync -C ./cherrybench

tmpdir="$(mktemp -d)"
./make_config.sh > "$tmpdir/config.toml"

venv="$(poetry show -C ./cherrybench -v | grep "Using virtualenv:" | cut -c19-)"
sudo "$venv/bin/python" -m cherrybench "$tmpdir/config.toml"