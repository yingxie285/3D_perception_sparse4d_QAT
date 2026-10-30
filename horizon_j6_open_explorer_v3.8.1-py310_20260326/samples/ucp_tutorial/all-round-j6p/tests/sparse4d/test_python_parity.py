#!/usr/bin/env python3
"""Compare compiled C++ decoder/corners with the actual repository Python functions.
This is a synthetic-output algorithm test, not an HBM accuracy test.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--decoder-exe", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    exe = args.decoder_exe.resolve()
    sys.path.insert(0, str(root))
    decoder = load_module(root / "modules/head/decoder/decoder.py", "repo_decoder")
    viz = load_module(root / "tool/visualization/utils.py", "repo_viz")
    rng = np.random.default_rng(20261028)
    logits = rng.normal(0, 1.2, (1, 384, 13)).astype(np.float32)
    boxes = rng.normal(0, .5, (1, 384, 8)).astype(np.float32)
    boxes[..., :3] *= 10
    for top_k, threshold in [(300, .5), (300, .97), (20, .9)]:
        ref = decoder.SparseBox3DDecoder(top_k, threshold).decode(
            [torch.from_numpy(logits)], [torch.from_numpy(boxes)], qulity=[None])[0]
        with tempfile.TemporaryDirectory(prefix="sparse4d_decoder_") as tmp:
            tmp = Path(tmp)
            (tmp / "input.json").write_text(json.dumps({
                "logits": logits.ravel().tolist(), "boxes": boxes.ravel().tolist(),
                "top_k": top_k, "threshold": threshold,
            }))
            subprocess.run([str(exe), "input.json", "output.json"], cwd=tmp, check=True)
            got = json.loads((tmp / "output.json").read_text())
        assert len(got) == len(ref["scores_3d"])
        np.testing.assert_array_equal([d["label"] for d in got], ref["labels_3d"].numpy())
        np.testing.assert_allclose([d["score"] for d in got], ref["scores_3d"].numpy(), rtol=2e-6, atol=1e-7)
        actual_boxes = np.asarray([d["box"] for d in got])
        np.testing.assert_allclose(actual_boxes, ref["boxes_3d"].numpy(), rtol=2e-6, atol=1e-6)
        # Use C++ decoded boxes as input to isolate the corner conversion.
        expected_corners = viz.box3d_to_corners(actual_boxes)
        np.testing.assert_allclose([d["corners"] for d in got], expected_corners, rtol=1e-12, atol=1e-12)
        print(f"PASS repo decoder + corners: top_k={top_k} threshold={threshold} detections={len(got)}")

if __name__ == "__main__":
    main()

