import os
import json

root_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/whut数据集/fisheye_data_2w_frame"
missing_file = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/tool/data_converter/缺失.txt"
all_frame_file = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/tool/data_converter/all_frame_ids.txt"

missing_frames = set()
valid_frames = []

total_files = 0

for subdir, dirs, files in os.walk(root_dir):
    if os.path.basename(subdir) == "result":
        for file in files:
            if not file.endswith(".json"):
                continue

            total_files += 1
            json_path = os.path.join(subdir, file)

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                objects = data.get("objects", [])

                # 若没有 object，按需求选择：跳过
                if len(objects) == 0:
                    continue

                # 默认：前视缺失
                frame_missing = True

                for obj in objects:
                    bbox2d = obj.get("2D_bbox", {})
                    if "image0" in bbox2d:
                        frame_missing = False
                        break

                frame_id = os.path.splitext(file)[0]

                if frame_missing:
                    missing_frames.add(frame_id)
                else:
                    valid_frames.append(frame_id)

            except Exception as e:
                print(f"读取失败: {json_path}, 错误: {e}")

# 写入缺失帧
with open(missing_file, "w") as f:
    for frame_id in sorted(missing_frames):
        f.write(frame_id + "\n")

# 写入有效帧（按顺序）
with open(all_frame_file, "w") as f:
    for frame_id in valid_frames:
        f.write(frame_id + "\n")

print("遍历 JSON 文件数量:", total_files)
print("前视(image0)完全缺失的帧数量:", len(missing_frames))
print("有效帧数量:", len(valid_frames))
print("缺失列表已写入:", missing_file)
print("全部有效帧ID已写入:", all_frame_file)
