"""Decode actual HBM dumps using the repository decoder, without model forward.

Writes a token-indexed subset pickle; full dataset.evaluate must use the same subset.
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--dump-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('sparse4d_decoder', args.repo_root / 'modules/head/decoder/decoder.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = sorted(args.dump_dir.rglob('detections.json'))
    if not files:
        raise ValueError('No completed HBM detection dumps found')
    results, tokens = [], []
    for file in files:
        meta = json.loads(file.read_text())
        token = meta['frame_id']
        if token in tokens:
            raise ValueError('Duplicate frame: ' + token)
        tokens.append(token)
        logits = np.fromfile(file.parent / 'classification_f32.bin', dtype='<f4').reshape(1, 384, 13)
        boxes = np.fromfile(file.parent / 'prediction_f32.bin', dtype='<f4').reshape(1, 384, 8)
        decoder = module.SparseBox3DDecoder(meta['top_k'], meta['score_threshold'])
        decoded = decoder.decode([torch.from_numpy(logits)], [torch.from_numpy(boxes)], qulity=[None])[0]
        actual = meta['detections']
        np.testing.assert_array_equal([x['label'] for x in actual], decoded['labels_3d'].numpy())
        np.testing.assert_allclose([x['score'] for x in actual], decoded['scores_3d'].numpy(), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(np.asarray([x['box_lidar_xyz_wlh_yaw'] for x in actual]).reshape(-1, 7), decoded['boxes_3d'].numpy(), rtol=2e-6, atol=1e-6)
        results.append({'img_bbox': decoded})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(results, f)
    args.output.with_suffix('.tokens.json').write_text(json.dumps(tokens, indent=2))
    print('HBM decoder parity passed:', len(results), 'frames. Match tokens to dataset subset before evaluation.')


if __name__ == '__main__':
    main()
