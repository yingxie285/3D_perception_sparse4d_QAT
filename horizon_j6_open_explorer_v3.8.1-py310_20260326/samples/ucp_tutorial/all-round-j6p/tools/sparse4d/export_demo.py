#!/usr/bin/env python3
"""Export a portable multi-frame J6M/J6P demo; standard library only.

Keep export_frame.py alongside this script. No model inference is performed.
"""
import argparse
import json
from pathlib import Path
import shutil

from export_frame import (load_config, load_records, preprocess_spec, matrix,
                          distortion, original_lidar2img, resolve_image, git_commit)

CAMERAS = ('FISHEYE_CAM_FRONT', 'FISHEYE_CAM_RIGHT', 'FISHEYE_CAM_LEFT', 'FISHEYE_CAM_BACK')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
    parser.add_argument('--ann-path', type=Path, required=True, help='One scene JSON or directory containing scene JSON files')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-frames', type=int, default=1, help='Total across all selected scenes; -1 means all remaining frames')
    parser.add_argument('--start-index', type=int, default=0, help='Zero-based index after global timestamp sorting')
    parser.add_argument('--stride', type=int, default=1, help='Select every Nth frame from start-index')
    args = parser.parse_args()
    if args.num_frames != -1 and args.num_frames <= 0:
        parser.error('--num-frames must be positive or -1')
    if args.start_index < 0 or args.stride <= 0:
        parser.error('start-index must be nonnegative and stride positive')
    repo = args.repo_root.resolve()
    config = load_config(args.config if args.config.is_absolute() else repo / args.config)
    files = sorted(args.ann_path.glob('*.json')) if args.ann_path.is_dir() else [args.ann_path]
    records = [(info, file) for file in files for info in load_records(file)]
    records.sort(key=lambda pair: pair[0]['timestamp'])
    selected = list(enumerate(records))[args.start_index::args.stride]
    if args.num_frames != -1:
        if len(selected) < args.num_frames:
            parser.error(f'Requested {args.num_frames} frames, but only {len(selected)} available after start/stride')
        selected = selected[:args.num_frames]
    if not selected:
        parser.error('No selected frames')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    # Resolve all selected images and calibration before copying any frame.
    prepared = []
    tokens = set()
    for source_index, (info, file) in selected:
        token = str(info['token'])
        if token in tokens:
            raise ValueError('Duplicate frame token: ' + token)
        tokens.add(token)
        if set(info['cams']) != set(CAMERAS):
            raise ValueError('Unexpected camera names for frame ' + token)
        cameras = []
        for index, name in enumerate(CAMERAS):
            cam = info['cams'][name]
            image = resolve_image(cam['data_path'], repo, Path(config['data']['val']['data_root']), args.image_root.resolve())
            k = matrix(cam['cam_intrinsic'], 3, 3, name)
            e = matrix(cam['lidar2camera_extrinsic'], 4, 4, name)
            if 'cam_extrinsic' in cam:
                alias = matrix(cam['cam_extrinsic'], 4, 4, name)
                if any(abs(alias[i][j] - e[i][j]) > 1e-8 for i in range(4) for j in range(4)):
                    raise ValueError('Conflicting extrinsic fields: ' + token + '/' + name)
            cameras.append((image, dict(name=name, image=f'images/{index:02d}_{name}{image.suffix.lower()}',
                                       cam_intrinsic=k, cam_extrinsic=e,
                                       cam_distortion=distortion(cam['distortion']), lidar2img=original_lidar2img(k, e))))
        prepared.append((source_index, info, file, cameras))
    output.mkdir(parents=True, exist_ok=True)
    paths, frames = [], []
    commit = git_commit(repo, None)
    for index, (source_index, info, file, cameras) in enumerate(prepared):
        relative = f'frames/frame_{index:06d}/frame.json'
        folder = (output / relative).parent
        (folder / 'images').mkdir(parents=True)
        for source, cam in cameras:
            shutil.copy2(source, folder / cam['image'])
        manifest = dict(schema_version=1, frame_id=str(info['token']), timestamp=info['timestamp'],
                        scene=file.stem, source_index=source_index, source_commit=commit,
                        cameras=[cam for _, cam in cameras], preprocess=preprocess_spec(config), time_interval=1.0)
        (output / relative).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        paths.append(relative)
        frames.append(dict(manifest=relative, frame_id=str(info['token']), scene=file.stem,
                           timestamp=info['timestamp'], source_index=source_index))
        print(f'[{index+1}/{len(prepared)}] {file.stem} {info["token"]}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(paths, indent=2), encoding='utf-8')
    summary = dict(num_frames=len(paths), camera_order=CAMERAS, start_index=args.start_index,
                   stride=args.stride, ordering='global timestamp ascending; no implicit dataset load_interval', frames=frames)
    (output / 'dataset.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Export complete: {output}\nBoard arguments: --manifest=<demo>/manifest.json --num_frames=-1')


if __name__ == '__main__':
    main()
