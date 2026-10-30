#!/usr/bin/env python3
"""Compare board tensor dumps against reference float32 tensor bins.

For each semantic, the candidate directory must contain ``<semantic>.bin`` in
contiguous float32 order.  An optional candidate manifest can name different
files, but both sides must declare the same tensor shape and dtype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference tensor manifest JSON")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--candidate-dir", type=Path)
    selector.add_argument("--candidate-inputs", type=Path, help="Candidate tensor JSON with semantic, shape, path")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("inputs", payload.get("outputs", payload.get("tensors")))
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected 'inputs', 'outputs', or 'tensors' list")
    result: dict[str, dict] = {}
    for item in items:
        semantic = item.get("semantic")
        shape = item.get("shape")
        relative = item.get("path")
        if not isinstance(semantic, str) or semantic in result:
            raise ValueError(f"{path}: duplicate or invalid semantic {semantic!r}")
        if not isinstance(shape, list) or any(not isinstance(v, int) or v < 0 for v in shape):
            raise ValueError(f"{path}: invalid shape for {semantic}")
        if item.get("dtype", "float32") != "float32" or not isinstance(relative, str):
            raise ValueError(f"{path}: {semantic} must have float32 dtype and file path")
        item = dict(item)
        item["resolved_path"] = (path.parent / relative).resolve()
        result[semantic] = item
    return result


def main() -> int:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be nonnegative")
    import numpy as np

    reference = load_manifest(args.reference.resolve())
    candidate = load_manifest(args.candidate_inputs.resolve()) if args.candidate_inputs else None
    if candidate is not None and set(candidate) != set(reference):
        raise ValueError("Semantic sets differ: " + str(sorted(set(reference) ^ set(candidate))))
    success = True
    for semantic, item in reference.items():
        ref_shape = tuple(item["shape"])
        ref_path = item["resolved_path"]
        if candidate is None:
            cand_path = args.candidate_dir.resolve() / f"{semantic}.bin"
            cand_shape = ref_shape
        else:
            cand_path = candidate[semantic]["resolved_path"]
            cand_shape = tuple(candidate[semantic]["shape"])
        if ref_shape != cand_shape:
            print(f"FAIL {semantic}: shape {ref_shape} versus {cand_shape}")
            success = False
            continue
        expected_count = int(np.prod(ref_shape, dtype=np.int64)) if ref_shape else 1
        if not ref_path.is_file() or not cand_path.is_file():
            print(f"FAIL {semantic}: missing {ref_path if not ref_path.is_file() else cand_path}")
            success = False
            continue
        ref = np.fromfile(ref_path, dtype="<f4")
        got = np.fromfile(cand_path, dtype="<f4")
        if len(ref) != expected_count or len(got) != expected_count:
            print(f"FAIL {semantic}: expected {expected_count} floats; found ref={len(ref)}, board={len(got)}")
            success = False
            continue
        same_nan = np.isnan(ref) & np.isnan(got)
        finite = np.isfinite(ref) & np.isfinite(got)
        error = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        allowed = args.atol + args.rtol * np.abs(ref.astype(np.float64))
        bad = ~(same_nan | (finite & (error <= allowed)))
        finite_error = error[finite]
        max_abs = float(finite_error.max()) if finite_error.size else float("nan")
        mean_abs = float(finite_error.mean()) if finite_error.size else float("nan")
        status = "PASS" if not bad.any() else "FAIL"
        print(f"{status} {semantic} shape={list(ref_shape)} bad={int(bad.sum())}/{expected_count} "
              f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
        if bad.any():
            success = False
            for flat_index in np.flatnonzero(bad)[: args.max_examples]:
                index = np.unravel_index(flat_index, ref_shape) if ref_shape else ()
                print(f"  index={index} ref={ref[flat_index]!r} board={got[flat_index]!r}")
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
