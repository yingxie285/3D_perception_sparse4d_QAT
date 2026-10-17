#由于前视视角存在没有目标的帧，2万帧里一共18852帧前视视角有目标，有1148帧没有目标，因此只生成18852帧的data_info.json

import os
import json

# =========================
# 路径配置
# =========================
dataset_root = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame'

train_val_split_path = os.path.join(
    dataset_root, 'train_val_split.json'
)

all_frame_txt = os.path.join(
    dataset_root, 'all_frame_ids.txt'   #这是前视视角有目标的帧，2万帧里一共18852帧前视视角有目标，只生成18852帧的data_info.json
)

output_json = os.path.join(
    dataset_root, 'data_info.json'
)

# =========================
# 读取“有效目标帧”列表
# =========================
with open(all_frame_txt, 'r') as f:
    valid_frame_ids = set(
        line.strip() for line in f if line.strip()
    )

print(f"[INFO] Loaded valid frame ids: {len(valid_frame_ids)}")

# =========================
# 读取 train / val 场景划分
# =========================
with open(train_val_split_path, 'r') as f:
    scene_split = json.load(f)

split_dict = {
    "train": [],
    "val": []
}

skipped_frames = 0

# =========================
# 构建 data_info
# =========================
for split in ['train', 'val']:
    for scene in scene_split[split]:

        pc_dir = os.path.join(scene, 'point_cloud')
        front_img_dir = os.path.join(scene, 'image0')
        right_img_dir = os.path.join(scene, 'image1')
        left_img_dir = os.path.join(scene, 'image2')
        back_img_dir = os.path.join(scene, 'image3')
        position_dir = os.path.join(scene, 'position')
        cam_cfg_dir = os.path.join(scene, 'camera_config')
        anno_dir = os.path.join(scene, 'result')

        pc_abs_path = os.path.join(dataset_root, pc_dir)

        if not os.path.isdir(pc_abs_path):
            continue

        # 获取该 scene 下所有帧（以点云为准）
        file_list = sorted([
            f for f in os.listdir(pc_abs_path)
            if f.endswith('.npy') and os.path.isfile(os.path.join(pc_abs_path, f))
        ])

        if not file_list:
            continue

        num_frames = len(file_list)
        start_frame_id = os.path.splitext(file_list[0])[0]
        end_frame_id = os.path.splitext(file_list[-1])[0]

        for fname in file_list:
            sample_id = os.path.splitext(fname)[0]

            # =========================
            # 核心：过滤无目标帧
            # =========================
            if sample_id not in valid_frame_ids:
                skipped_frames += 1
                continue

            pc_path = os.path.join(pc_dir, fname)

            sample = {
                "sequence_id": scene,
                "frame_id": sample_id,
                "pointcloud_path": pc_path,
                "front_image_path": os.path.join(front_img_dir, sample_id + ".jpeg"),
                "right_image_path": os.path.join(right_img_dir, sample_id + ".jpeg"),
                "left_image_path": os.path.join(left_img_dir, sample_id + ".jpeg"),
                "back_image_path": os.path.join(back_img_dir, sample_id + ".jpeg"),
                "ego_pose_path": os.path.join(position_dir, sample_id + ".json"),
                "cam_cali_path": os.path.join(cam_cfg_dir, sample_id + ".json"),
                "anno_path": os.path.join(anno_dir, sample_id + ".json"),
                "timestamp": sample_id,
                "num_frames": num_frames,
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id
            }

            split_dict[split].append(sample)

# =========================
# 统计信息
# =========================
split_dict["train_frames"] = len(split_dict["train"])
split_dict["val_frames"] = len(split_dict["val"])
print(split_dict["val"])
# =========================
# 导出 val 帧 id
# =========================
val_txt_path = os.path.join(dataset_root, 'val.txt')

with open(val_txt_path, 'w') as f:
    for sample in split_dict["val"]:
        f.write(sample["frame_id"] + '\n')

print(f"[INFO] val.txt written successfully! Total: {len(split_dict['val'])}")

# =========================
# 写入 JSON
# =========================
with open(output_json, 'w') as f:
    json.dump(split_dict, f, indent=4)

# =========================
# 打印日志
# =========================
print("[INFO] data_info.json written successfully!")
print("[INFO] Train frames:", split_dict["train_frames"])
print("[INFO] Val frames:", split_dict["val_frames"])
print("[INFO] Skipped empty frames:", skipped_frames)


#以下是不去除任何样本，采用2万帧样本
# import os
# import random
# import json

# # 数据集路径
# dataset_root = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame'

# # 先读取已有的场景划分 JSON（tool/data_converter/get_train_val.py生成的train_val_split.json）
# with open('/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame/train_val_split.json', 'r') as f:
#     scene_split = json.load(f)
# split_dict = {"train": [], "val": []}
# for split in ['train', 'val']:
#     for scene in scene_split[split]:
#         pc_dir = os.path.join(scene, 'point_cloud')
#         front_img_dir = os.path.join(scene, 'image0')
#         right_img_dir = os.path.join(scene, 'image1')
#         left_img_dir = os.path.join(scene, 'image2')
#         back_img_dir = os.path.join(scene, 'image3')
#         position_dir = os.path.join(scene, 'position')
#         cam_cfg_dir = os.path.join(scene, 'camera_config')
#         anno_dir = os.path.join(scene, 'result')
#         pc_abs_path = os.path.join(dataset_root, pc_dir)

#         # 获取所有 .npy 文件名（已排序）
#         file_list = sorted([
#             f for f in os.listdir(pc_abs_path)
#             if f.endswith('.npy') and os.path.isfile(os.path.join(pc_abs_path, f))  #1216
#         ])
#         if not file_list:
#             continue  # 跳过空 scene
#         num_frames = len(file_list)
#         start_frame_id = os.path.splitext(file_list[0])[0]
#         end_frame_id = os.path.splitext(file_list[-1])[0]

#         for fname in file_list:
#             sample_id = os.path.splitext(fname)[0]
#             pc_path = os.path.join(pc_dir, fname)
#             front_img_path = os.path.join(front_img_dir, sample_id + ".jpeg")
#             right_img_path = os.path.join(right_img_dir, sample_id + ".jpeg")
#             left_img_path = os.path.join(left_img_dir, sample_id + ".jpeg")
#             back_img_path = os.path.join(back_img_dir, sample_id + ".jpeg")
#             position_path = os.path.join(position_dir, sample_id + ".json")
#             cam_cfg_path = os.path.join(cam_cfg_dir, sample_id + ".json")
#             anno_path = os.path.join(anno_dir, sample_id + ".json")

#             sample = {
#                 "sequence_id": scene,
#                 "frame_id": sample_id,
#                 "pointcloud_path": pc_path,
#                 "front_image_path": front_img_path,
#                 "right_image_path": right_img_path,
#                 "left_image_path": left_img_path,
#                 "back_image_path": back_img_path,
#                 "ego_pose_path": position_path,
#                 "cam_cali_path": cam_cfg_path,
#                 "anno_path": anno_path,
#                 "timestamp": sample_id,
#                 "num_frames": num_frames,
#                 "start_frame_id": start_frame_id,
#                 "end_frame_id": end_frame_id
#             }
#             split_dict[split].append(sample)

# # 统计帧数
# train_frames = len(split_dict["train"])
# val_frames = len(split_dict["val"])

# # 将统计结果加入 JSON 中
# split_dict["train_frames"] = train_frames
# split_dict["val_frames"] = val_frames


# with open('/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame/data_info.json', 'w') as f:
#     json.dump(split_dict, f, indent=4)

# print("写入完成!")
# print("Train frames:", train_frames)
# print("Val frames:", val_frames)
