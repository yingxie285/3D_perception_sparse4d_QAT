#!/usr/bin/env python3
"""Export one validation frame and, optionally, its exact Python pipeline inputs.

The normal manifest path uses only the Python standard library.  --dump-reference
imports the project's numpy/torch/OpenCV/Pillow pipeline on the training server.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import pickle
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SEMANTICS = (
    "img",
    "cam_extrinsic",
    "cam_intrinsic",
    "cam_distortion",
    "lidar2img",
    "image_wh",
    "time_interval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--info-file", type=Path,
        help="Override data.val.ann_file; a JSON/PKL file or a directory of such files",
    )
    parser.add_argument(
        "--image-root", type=Path,
        help="Alternate root for paths stored in the annotation file",
    )
    parser.add_argument("--source-commit", help="Override git HEAD when .git is absent")
    parser.add_argument(
        "--dump-reference", action="store_true",
        help="Run the repository's test transforms and save seven float32 input bins",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    # The Sparse4D config is a plain Python file; runpy does not import torch.
    config = runpy.run_path(str(path))
    if not isinstance(config.get("data"), dict):
        raise ValueError(f"{path}: missing data dictionary")
    return config


def as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [as_list(item) for item in value]
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Calibration contains a non-finite number")
        return result
    raise TypeError(f"Unsupported calibration value: {type(value).__name__}")


def matrix(value: Any, rows: int, cols: int, name: str) -> list[list[float]]:
    value = as_list(value)
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a matrix")
    if len(value) == rows * cols and all(isinstance(v, float) for v in value):
        value = [value[i * cols : (i + 1) * cols] for i in range(rows)]
    if len(value) != rows or any(not isinstance(v, list) or len(v) != cols for v in value):
        raise ValueError(f"{name}: expected shape {rows}x{cols}")
    return value


def distortion(value: Any) -> list[float]:
    value = as_list(value)
    def flatten(obj: Any) -> list[float]:
        if isinstance(obj, list):
            return [coefficient for item in obj for coefficient in flatten(item)]
        return [obj]
    result = flatten(value)
    if len(result) != 4 or not all(isinstance(v, float) for v in result):
        raise ValueError("cam_distortion must have exactly four fisheye coefficients")
    return result


def matmul_4x4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def original_lidar2img(k: list[list[float]], e: list[list[float]]) -> list[list[float]]:
    pad = [[float(i == j) for j in range(4)] for i in range(4)]
    for i in range(3):
        for j in range(3):
            pad[i][j] = k[i][j]
    return matmul_4x4(pad, e)


def records_from_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("infos", "data_list", "samples", "annotations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
        raise ValueError(f"{source}: expected a list of frame dictionaries")
    return payload


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".json", ".pkl", ".pickle"))
        if not files:
            raise ValueError(f"No JSON/PKL annotation files in {path}")
        records: list[dict[str, Any]] = []
        for file in files:
            records.extend(load_records(file))
        return records
    if not path.is_file():
        raise FileNotFoundError(f"Annotation source does not exist: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif path.suffix.lower() in (".pkl", ".pickle"):
        # Pickle is executable data; use only locally trusted annotation files.
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported annotation format: {path}")
    return records_from_payload(payload, path)


def select_record(records: list[dict[str, Any]], index: int, interval: int) -> dict[str, Any]:
    # Matches NuScenes4DDetTrackDataset.load_annotations: timestamp sort, then stride.
    ordered = sorted(records, key=lambda x: x["timestamp"])[::interval]
    if index < 0 or index >= len(ordered):
        raise IndexError(f"index {index} outside validation dataset of {len(ordered)} frames")
    return ordered[index]


def resolve_image(path_text: str, repo_root: Path, data_root: Path | None, image_root: Path | None) -> Path:
    original = Path(path_text)
    candidates: list[Path] = []
    if image_root is not None:
        # Accept the user's scene directory, dataset root, or repository root.
        for index, part in enumerate(original.parts):
            if part == image_root.name:
                candidates.append(image_root.joinpath(*original.parts[index + 1:]))
        if original.is_absolute() and data_root is not None:
            try:
                candidates.append(image_root / original.relative_to(data_root))
            except ValueError:
                pass
        if not original.is_absolute():
            candidates.append(image_root / original)
        candidates.append(image_root / original.name)
    candidates.append(original)
    if not original.is_absolute():
        candidates.append(repo_root / original)
        if data_root is not None:
            candidates.append(data_root / original)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Image {path_text!r} not found. Tried: {tried}")


def git_commit(repo_root: Path, override: str | None) -> str:
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def preprocess_spec(config: dict[str, Any]) -> dict[str, Any]:
    aug = config["data"]["val"]["data_aug_conf"]
    image = config["img_norm_cfg"]
    source_h, source_w = int(aug["H"]), int(aug["W"])
    target_h, target_w = (int(v) for v in aug["final_dim"])
    resize = max(target_h / source_h, target_w / source_w)
    nominal_w, nominal_h = int(source_w * resize), int(source_h * resize)
    crop_h = int((1 - sum(aug["bot_pct_lim"]) / len(aug["bot_pct_lim"])) * nominal_h) - target_h
    crop_w = int(max(0, nominal_w - target_w) / 2)
    return {
        "resize": resize,
        # Dataset get_augmentation computes this from config H/W, but
        # ResizeCropFlipImage recomputes resize dimensions from each real image.
        "nominal_resize_dims": [nominal_w, nominal_h],
        "crop": [crop_w, crop_h, crop_w + target_w, crop_h + target_h],
        "mean": [float(v) for v in image["mean"]],
        "std": [float(v) for v in image["std"]],
        "to_rgb": bool(image["to_rgb"]),
        "flip": False,
        "rotate": 0,
        "config_source_size": [source_w, source_h],
    }


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def write_reference(
    repo_root: Path, config: dict[str, Any], manifest: dict[str, Any], output_dir: Path,
) -> None:
    # Import only here.  This executes the exact repository test transforms,
    # including its known camera-intrinsic scaling convention.
    import numpy as np

    sys.path.insert(0, str(repo_root))
    from dataset.pipeline.nuscenes import nusc_pipeline as pipeline  # type: ignore

    data: dict[str, Any] = {
        # Relative ASCII names avoid Windows OpenCV's Unicode-path limitation.
        "img_filename": [cam["image"] for cam in manifest["cameras"]],
        "cam_intrinsic": [np.asarray(cam["cam_intrinsic"], dtype=np.float64) for cam in manifest["cameras"]],
        "cam_extrinsic": [np.asarray(cam["cam_extrinsic"], dtype=np.float64) for cam in manifest["cameras"]],
        "cam_distortion": [np.asarray(cam["cam_distortion"], dtype=np.float64) for cam in manifest["cameras"]],
        "lidar2img": [np.asarray(cam["lidar2img"], dtype=np.float64) for cam in manifest["cameras"]],
        "aug_config": {
            "resize": manifest["preprocess"]["resize"],
            "crop": tuple(manifest["preprocess"]["crop"]),
            "flip": False,
            "rotate": 0,
        },
    }
    source_shapes: list[list[int]] | None = None
    # The validation config currently uses the same test_pipeline object.
    for transform_spec in config["data"]["val"]["pipeline"]:
        name = transform_spec["type"]
        if name == "Collect":
            break  # input values are already finalized by the adaptor
        if name not in (
            "LoadMultiViewImageFromFiles", "ResizeCropFlipImage",
            "NormalizeMultiviewImage", "NuScenesSparse4DAdaptor",
        ):
            raise ValueError(f"Reference export cannot silently skip transform {name}")
        kwargs = {k: v for k, v in transform_spec.items() if k != "type"}
        with working_directory(output_dir):
            data = getattr(pipeline, name)(**kwargs)(data)
        if name == "LoadMultiViewImageFromFiles":
            source_shapes = [list(image.shape[:2]) for image in data["img"]]

    def array(value: Any) -> Any:
        value = getattr(value, "data", value)  # DataContainer
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.ascontiguousarray(value, dtype=np.float32)

    values = {
        "img": array(data["img"])[None],
        "cam_extrinsic": array(data["cam_extrinsic"])[None],
        "cam_intrinsic": array(data["cam_intrinsic"])[None],
        "cam_distortion": array(data["cam_distortion"])[None],
        "lidar2img": array(data["lidar2img"])[None],
        "image_wh": array(data["image_wh"])[None],
        "time_interval": np.asarray(manifest["time_interval"], dtype=np.float32),
    }
    reference_dir = output_dir / "reference_inputs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for semantic in SEMANTICS:
        value = np.ascontiguousarray(values[semantic], dtype=np.float32)
        relative = Path("reference_inputs") / f"{semantic}.bin"
        value.tofile(output_dir / relative)
        items.append({
            "semantic": semantic, "dtype": "float32", "shape": list(value.shape),
            "path": relative.as_posix(),
        })
    payload = {
        "schema_version": 1,
        "frame_id": manifest["frame_id"],
        "source_commit": manifest["source_commit"],
        "source_image_shapes_hw": source_shapes,
        "inputs": items,
    }
    (output_dir / "reference_inputs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    if source_shapes and any([shape[1], shape[0]] != manifest["preprocess"]["config_source_size"] for shape in source_shapes):
        print(
            "WARNING: real image dimensions differ from config H/W. The test "
            "pipeline still uses config-derived crop but resizes each actual image.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_config(config_path.resolve())
    source = args.info_file or Path(config["data"]["val"]["ann_file"])
    if not source.is_absolute():
        source = repo_root / source
    records = load_records(source.resolve())
    interval = int(config["data"]["val"].get("load_interval", 1))
    if interval < 1:
        raise ValueError("load_interval must be positive")
    info = select_record(records, args.index, interval)
    cams = info.get("cams")
    if not isinstance(cams, dict) or len(cams) != 4:
        raise ValueError("Expected exactly four cameras in info['cams']")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data"]["val"]["data_root"])
    image_root = args.image_root.resolve() if args.image_root else None
    cameras = []
    for number, (name, cam_info) in enumerate(cams.items()):
        if not isinstance(cam_info, dict):
            raise ValueError(f"Camera {name}: expected a dictionary")
        raw = resolve_image(str(cam_info["data_path"]), repo_root, data_root, image_root)
        filename = f"{number:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)}{raw.suffix.lower()}"
        relative = Path("images") / filename
        shutil.copy2(raw, output_dir / relative)
        k = matrix(cam_info["cam_intrinsic"], 3, 3, f"{name}.cam_intrinsic")
        e = matrix(cam_info["lidar2camera_extrinsic"], 4, 4, f"{name}.cam_extrinsic")
        if "cam_extrinsic" in cam_info:
            alias = matrix(cam_info["cam_extrinsic"], 4, 4, f"{name}.cam_extrinsic")
            if any(abs(alias[i][j] - e[i][j]) > 1e-8 for i in range(4) for j in range(4)):
                raise ValueError(f"{name}: cam_extrinsic differs from lidar2camera_extrinsic; "
                                 "the repository uses lidar2camera_extrinsic, resolve this conflict first")
        cameras.append({
            "name": name,
            "image": relative.as_posix(),
            "source_data_path": str(cam_info["data_path"]),
            "cam_intrinsic": k,
            "cam_extrinsic": e,
            "cam_distortion": distortion(cam_info["distortion"]),
            "lidar2img": original_lidar2img(k, e),
        })
    manifest = {
        "schema_version": 1,
        "frame_id": str(info.get("token", f"index_{args.index}")),
        "source_commit": git_commit(repo_root, args.source_commit),
        "cameras": cameras,
        "preprocess": preprocess_spec(config),
        "time_interval": 1.0,
    }
    (output_dir / "frame.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    if args.dump_reference:
        write_reference(repo_root, config, manifest, output_dir)
    print(f"Exported frame {manifest['frame_id']} to {output_dir}")
    print("Camera order:", ", ".join(cam["name"] for cam in cameras))
    if manifest["source_commit"] == "unknown":
        print("WARNING: source commit unknown; use --source-commit if needed", file=sys.stderr)


if __name__ == "__main__":
    main()
