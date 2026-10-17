import os
import json
from collections import Counter

val_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/BEVOD_nusc_anno_dumpjson_mini/val"

cls_counter = Counter()
all_classes = set()

for fname in os.listdir(val_dir):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(val_dir, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    # 关键：处理最外层 list
    if isinstance(data, list):
        if len(data) == 0:
            print(f"[WARN] {fname} is empty list")
            continue
        data = data[0]

    if "gt_names" not in data:
        print(f"[WARN] {fname} has no 'gt_names' key")
        continue

    gt_names = data["gt_names"]
    all_classes.update(gt_names)
    cls_counter.update(gt_names)

print("\n=== 唯一类别 ===")
print(sorted(all_classes))

print("\n=== 各类别数量统计 ===")
for k, v in cls_counter.items():
    print(f"{k}: {v}")
