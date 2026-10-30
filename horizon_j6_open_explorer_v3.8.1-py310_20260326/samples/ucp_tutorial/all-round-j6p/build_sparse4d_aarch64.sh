#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LINARO_GCC_ROOT:?Set LINARO_GCC_ROOT to the aarch64 toolchain root}"
export CC="$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-gcc"
export CXX="$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++"
cmake -S "$root" -B "$root/build_sparse4d_aarch64" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DSPARSE4D_ARCH=aarch64 -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build_sparse4d_aarch64" --parallel "${JOBS:-8}"
cmake --build "$root/build_sparse4d_aarch64" --target install
