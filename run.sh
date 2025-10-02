#!/usr/bin/env bash
set -e

# Parse arguments to separate --avx512 for make_config.sh from other args for cherrybench
make_config_args=()
cherrybench_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --avx512|--no-timeout)
            make_config_args+=("$1")
            shift
            ;;
        *)
            cherrybench_args+=("$1")
            shift
            ;;
    esac
done

poetry sync -C ./cherrybench

tmpdir="$(mktemp -d)"
./make_config.sh "${make_config_args[@]}" > "$tmpdir/config.toml"

venv="$(poetry show -C ./cherrybench -v 2>&1 | grep --color=never "Using virtualenv:" | cut -c19-)"
sudo "$venv/bin/python" -m cherrybench "$tmpdir/config.toml" "${cherrybench_args[@]}"