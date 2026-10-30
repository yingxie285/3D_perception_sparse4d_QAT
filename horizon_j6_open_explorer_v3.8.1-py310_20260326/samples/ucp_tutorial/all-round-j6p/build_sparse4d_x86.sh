#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$root" -B "$root/build_sparse4d_x86" -DSPARSE4D_ARCH=x86 -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build_sparse4d_x86" --parallel "${JOBS:-8}"
cmake --build "$root/build_sparse4d_x86" --target install
