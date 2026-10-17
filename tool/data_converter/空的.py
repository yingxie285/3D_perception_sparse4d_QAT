import os
import json

root_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/whut数据集/fisheye_data_2w_frame"

total_files = 0
no_contour_count = 0
empty_size3d_count = 0

for subdir, dirs, files in os.walk(root_dir):
    if os.path.basename(subdir) == "result":
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                total_files += 1

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    for obj in data.get("objects", []):

                        # 情况1：没有 contour
                        if "contour" not in obj:
                            no_contour_count += 1
                            continue

                        contour = obj["contour"]

                        # 情况2：没有 size3D 或 size3D 为空
                        size3d = contour.get("size3D")
                        if (
                            size3d is None or
                            not isinstance(size3d, dict) or
                            len(size3d) == 0
                        ):
                            empty_size3d_count += 1

                except Exception as e:
                    print(f"读取失败: {json_path}, 错误: {e}")

print("遍历 JSON 文件数量:", total_files)
print("没有 contour 的 object 数量:", no_contour_count)
print("size3D 为空或不存在的 object 数量:", empty_size3d_count)