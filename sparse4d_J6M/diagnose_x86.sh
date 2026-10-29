#!/usr/bin/env bash
# Diagnostic budget, not expected model latency.
set -euo pipefail
sample_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
limit="${LIMIT_SECONDS:-600}"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT_SECONDS must be positive" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }
out="$sample_dir/output_diagnostic"
mkdir -p "$out"
echo "One frame; trace enabled; external limit ${limit}s. Stop any older inference first."
set +e
timeout --signal=TERM --kill-after=10s "${limit}s" \
  bash "$sample_dir/run_x86.sh" \
  --model_file "$sample_dir/assets/model.hbm" \
  --frame_config "$sample_dir/assets/frame_000/frame.json" \
  --output_dir "$out" --dump_inputs --trace --warmup 0 --repeat 1 "$@" \
  2>&1 | tee "$out/run.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" == 124 || "$status" == 137 ]]; then
  echo "Time limit reached. Keep run.log; final ENTER without RETURN identifies the stage." >&2
  echo "A timeout alone does NOT distinguish slow simulation from deadlock." >&2
fi
exit "$status"

