import json
from collections import Counter

json_path = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/e2e_worklog_test0120/sparse4dv3_temporal_r50_1x8_bs6_256x704/eval/img_bbox/results_nusc.json"

with open(json_path, "r") as f:
    data = json.load(f)

assert "results" in data, "results_nusc.json 中未找到 results 键"

cls_counter = Counter()
total_boxes = 0

for sample_token, det_list in data["results"].items():
    if not isinstance(det_list, list):
        continue
    for det in det_list:
        name = det.get("detection_name", None)
        if name is not None:
            cls_counter[name] += 1
            total_boxes += 1

print("=== 预测唯一类别 ===")
print(sorted(cls_counter.keys()))

print("\n=== 各类别预测框数量 ===")
for k, v in cls_counter.most_common():
    print(f"{k}: {v}")

print(f"\n=== 总预测框数 === {total_boxes}")
