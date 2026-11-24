#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
root="${LIBERO_DATA_ROOT:-$PWD/data/libero}"
mkdir -p "$root"
archive=libero_spatial_no_noops_lerobot.tar.gz
hf download yuanty/LIBERO-fastwam "$archive" --repo-type dataset --local-dir "$root"
tar -xzf "$root/$archive" -C "$root"
LIBERO_DATA_ROOT="$root" python tools/check_spatial_dataset.py
