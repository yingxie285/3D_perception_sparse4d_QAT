import os
import json
import hashlib

root_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame"
output_path = os.path.join(root_dir, "all_camera_config.json")

def json_hash(data):
    """
    对json内容做稳定hash，用于去重
    """
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()

unique_configs = {}
hash2id = {}
idx = 1

for seq_name in sorted(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue

    cam_cfg_dir = os.path.join(seq_path, "camera_config")
    if not os.path.isdir(cam_cfg_dir):
        continue

    for fname in sorted(os.listdir(cam_cfg_dir)):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(cam_cfg_dir, fname)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过损坏文件: {json_path}, error={e}")
            continue

        h = json_hash(data)
        if h in hash2id:
            # 重复，直接跳过
            continue

        key = str(idx)
        unique_configs[key] = data
        hash2id[h] = key
        idx += 1

with open(output_path, "w") as f:
    json.dump(unique_configs, f, indent=4)

print(f"完成，共发现 {len(unique_configs)} 套唯一 camera_config")
print(f"结果已写入: {output_path}")
