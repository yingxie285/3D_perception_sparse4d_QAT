#!/usr/bin/env bash
set -euo pipefail
bin="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_UCP_SIM_PLATFORM_TYPE=nash-m
exec "$bin/sparse4d_J6M" "$@"
