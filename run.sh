#!/usr/bin/env bash
set -e

poetry sync -C ./cherrybench

tmpdir="$(mktemp -d)"
./make_config.sh > "$tmpdir/config.toml"

venv="$(poetry show -C ./cherrybench -v 2>&1 | grep --color=never "Using virtualenv:" | cut -c19-)"
sudo "$venv/bin/python" -m cherrybench "$tmpdir/config.toml"