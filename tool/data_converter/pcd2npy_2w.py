import os
import numpy as np
from tqdm import tqdm


VERBOSE_FILE = False  # True: 每个 pcd 都打印；False: 只打印序列级


def load_pcd_xyz(pcd_path, load_dim=3):
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('DATA'):
                assert 'binary' in line.lower()
                break
            header.append(line)

        fields = sizes = types = None
        points_count = None
        for line in header:
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes = list(map(int, line.split()[1:]))
            elif line.startswith('TYPE'):
                types = line.split()[1:]
            elif line.startswith('POINTS'):
                points_count = int(line.split()[1])

        dtype = []
        for f_, s_, t_ in zip(fields, sizes, types):
            if t_ == 'F':
                dtype.append((f_, np.float32 if s_ == 4 else np.float64))
            elif t_ == 'U':
                dtype.append((f_, np.uint8 if s_ == 1 else np.uint16))
            elif t_ == 'I':
                dtype.append((f_, np.int32))

        data = np.fromfile(f, dtype=dtype)

        if points_count is None or len(data) != points_count:
            points_count = len(data)

        xyz = np.zeros((points_count, load_dim), dtype=np.float32)
        for i in range(load_dim):
            xyz[:, i] = data[fields[i]].astype(np.float32)

        return xyz


def convert_pcd_dir_inplace(point_cloud_dir, load_dim=3):
    pcd_files = sorted(f for f in os.listdir(point_cloud_dir) if f.endswith('.pcd'))
    if not pcd_files:
        return 0

    converted = 0
    for fname in pcd_files:
        pcd_path = os.path.join(point_cloud_dir, fname)
        npy_path = os.path.join(point_cloud_dir, fname.replace('.pcd', '.npy'))

        if os.path.exists(npy_path):
            continue

        if VERBOSE_FILE:
            print(f'    converting {fname} -> {fname.replace(".pcd", ".npy")}')

        pts = load_pcd_xyz(pcd_path, load_dim)
        np.save(npy_path, pts)
        converted += 1

    return converted


def convert_whole_fisheye_dataset(root_dir, load_dim=3):
    seq_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    print(f'[INFO] found {len(seq_dirs)} sequences\n')

    for seq in tqdm(seq_dirs, desc='Sequences'):
        print(f'\n[START] converting: {seq}')
        seq_path = os.path.join(root_dir, seq)
        pc_dir = os.path.join(seq_path, 'point_cloud')

        if not os.path.isdir(pc_dir):
            print(f'[SKIP]  no point_cloud dir')
            continue

        num = convert_pcd_dir_inplace(pc_dir, load_dim)

        print(f'[DONE]  converted {num} files in {seq}')


if __name__ == '__main__':
    convert_whole_fisheye_dataset(
        root_dir='/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/whut数据集/fisheye_data_2w_frame',
        load_dim=3
    )
