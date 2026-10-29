#!/usr/bin/env bash
# Vendor tool with exactly the quantized padded buffers dumped by the demo.
set -euo pipefail
sample_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${HRT_MODEL_EXEC:?Set HRT_MODEL_EXEC to the matching SDK x86 hrt_model_exec executable}"
[[ -x "$HRT_MODEL_EXEC" ]] || { echo "HRT_MODEL_EXEC is not executable" >&2; exit 2; }
limit="${LIMIT_SECONDS:-600}"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT_SECONDS must be positive" >&2; exit 2; }
runtime_root="${SPARSE4D_RUNTIME_ROOT:-$sample_dir/../../runtime/script_x86}"
export LD_LIBRARY_PATH="$runtime_root/x86/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_UCP_SIM_PLATFORM_TYPE=nash-m
inputs=""
for i in 0 1 2 3 4 5 6; do
  file="$sample_dir/output_diagnostic/packed_inputs/input_$i.bin"
  [[ -s "$file" ]] || { echo "Missing $file. Diagnose first; packing must finish." >&2; exit 2; }
  inputs="${inputs:+$inputs,}$file"
done
out="$sample_dir/output_hrt_diagnostic"
mkdir -p "$out"
set +e
timeout --signal=TERM --kill-after=10s "${limit}s" \
  "$HRT_MODEL_EXEC" infer \
  --model_file="$sample_dir/assets/model.hbm" --model_name=forward \
  --input_file="$inputs" --core_id=0 --frame_count=1 \
  --enable_dump=true --dump_format=bin --dump_path="$out" \
  --enable_cls_post_process=false \
  2>&1 | tee "$out/run.log"
status=${PIPESTATUS[0]}
set -e
exit "$status"

