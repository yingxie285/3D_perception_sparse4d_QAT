# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import json
import numpy as np
import pickle

from tqdm import tqdm
from pyquaternion import Quaternion



NameMapping = {
    "barrier": "barrier",
    "bus": "bus",
    "car": "car",
    "construction_vehicle": "construction_vehicle",
    "motor": "motor",
    "bicycle":"bicycle",
    "animal":"animal",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
    "stopper": "stopper",
    "trash_bin":"trash_bin",
    "sign":"sign",
    "truck": "truck",
}


class MyNuScenes:
    def __init__(self, data_root):
        self.data_root = data_root
        self._tables = {}
        self._load_all_tables()

    def _load_all_tables(self):
        """加载所有核心 json 表"""
        for name in [
            "scene", "sample", "sample_data", "ego_pose",
            "calibrated_sensor", "sample_annotation", "instance","sensor"
        ]:
            path = os.path.join(self.data_root, f"{name}.json")
            with open(path, 'r') as f:
                self._tables[name] = json.load(f)
        # 建立token->record索引，加快get操作
        self._token_index = {
            name: {item["token"]: item for item in self._tables[name]}
            for name in self._tables
        }

    def get(self, table_name, token):
        """模拟 nusc.get(table_name, token)"""
        return self._token_index[table_name][token]

    def get_sample_data_path(self, token):
        """获取对应 token 的点云文件路径，如果有多条记录则筛选npy 文件"""
    # sample_data 表是列表，不能用 self.get，因为 get 只返回第一条对应 token 的记录
        records = [sd for sd in self._tables["sample_data"] if sd["token"] == token]
        for sd in records:
            if sd["filename"].endswith(".npy") or "point_cloud" in sd["filename"]: #1216
                new_root = self.data_root  
                return os.path.join(new_root, sd["filename"])
        raise ValueError(f"No point cloud file found for token {token}")   

    def getind(self, table_name, token):
        """根据 token 获取索引编号（如果需要）"""
        for i, item in enumerate(self._tables[table_name]):
            if item["token"] == token:
                return i
        return -1

    @property
    def scene(self):
        return self._tables["scene"]

def load_fisheye_infos(pkl_path):
    """加载 .pkl 文件并提取 fisheye_infos"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']

def convert_numpy(obj):
    """将 NumPy 数组转换为 Python 列表以支持 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def obtain_sensor2top(
    nusc, sensor_token, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    data_path =nusc.get_sample_data_path(sensor_token)
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])  

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "lidar2camera_extrinsic": cs_record["camera_extrinsic"],
        "camera_intrinsic": cs_record["camera_intrinsic"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }


    return sweep


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    trainscene_nusc_infos = dict()
    valscene_nusc_infos = dict()
    for t in train_scenes:
        trainscene_nusc_infos[t] = list()
    for t in val_scenes:
        valscene_nusc_infos[t] = list()

    for sample in tqdm(nusc.sample, desc="Processing Sample"):
        lidar_token = sample["data"]["LIDAR"]
        # sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _, _, _ = nusc.get_sample_data(lidar_token)

        if not os.path.isfile(lidar_path):
            raise FileNotFoundError('file "{}" does not exist'.format(lidar_path))

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            # "lidar2ego_translation": cs_record["translation"],  # listlen=3
            # "lidar2ego_rotation": cs_record["rotation"],  # listlen=4
            "lidar2ego_translation": [],  # listlen=3
            "lidar2ego_rotation": [],  # listlen=4
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix  # numpy(3,3)
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix  # numpy(3,3)

        # obtain 6 image's information per frame
        camera_types = [
            "FISHEYE_CAM_FRONT",
            "FISHEYE_CAM_RIGHT",
            "FISHEYE_CAM_LEFT",
            "FISHEYE_CAM_BACK"
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic, cam_extrinsic, distortion = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, cam
            )
            # 将当前帧、当前相机的特定内外参更新进去
            cam_info.update(cam_intrinsic=cam_intrinsic.tolist(),
                            cam_extrinsic=cam_extrinsic.tolist(),
                            distortion=distortion.tolist())
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc,
                    sd_rec["prev"],
                    # l2e_t,
                    # l2e_r_mat,
                    # e2g_t,
                    # e2g_r_mat,
                    "lidar",
                )  # dict
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            # velocity = np.array(
            #     [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            # )
            valid_flag = (
                np.array(
                    [
                        (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                        for anno in annotations
                    ],
                    dtype=np.int32,
                )
                .reshape(-1)
                .tolist()
            )
            # convert velo from global to lidar
            # for i in range(len(boxes)):
            #     velo = np.array([*velocity[i], 0.0])
            #     velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            #     velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            # names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            # info["instance_inds"] = np.array(
            #     [
            #         nusc.getind("instance", x["instance_token"])
            #         for x in annotations
            #     ]
            # )

            info["instance_inds"] = [
                nusc.getind("instance", x["instance_token"]) for x in annotations
            ]

            info["gt_boxes"] = gt_boxes.tolist()
            info["gt_names"] = names
            # info["gt_velocity"] = velocity.reshape(-1, 2).tolist()
            info["num_lidar_pts"] = [a["num_lidar_pts"] for a in annotations]

            info["num_radar_pts"] = [a["num_radar_pts"] for a in annotations]
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            info.update(scene_token=sample["scene_token"])
            trainscene_nusc_infos[sample["scene_token"]].append(info)
            # train_nusc_infos.append(info)
        else:
            info.update(scene_token=sample["scene_token"])
            valscene_nusc_infos[sample["scene_token"]].append(info)
            # val_nusc_infos.append(info)

    return trainscene_nusc_infos, valscene_nusc_infos


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.
    nusc.scene:
        'token'
        'log_token'
        'nbr_samples'
        'first_sample_token'
        'last_sample_token'
        'name' : str="scene-0061"
        'description'
    nusc.get("sample", token)
        'token'
        'timestamp' : 1532402927647951
        'prev'
        'next'
        'scene_token'
        'data': {} 记录各个传感器当前帧率数据对应的token
                'RADAR_FRONT'
                'RADAR_FRONT_LEFT'
                'RADAR_FRONT_RIGHT'
                'RADAR_BACK_LEFT'
                'RADAR_BACK_RIGHT'
                'LIDAR_TOP'
                'CAM_FRONT'
                'CAM_FRONT_RIGHT'
                'CAM_BACK_RIGHT'
                'CAM_BACK'
                'CAM_BACK_LEFT'
                'CAM_FRONT_LEFT'
        'anns'
    nusc.get("sample_data", token): 存储传感器关键帧内外参和pose, 同时记录前一帧和后一帧非关键帧的token
                                    nusc.get_sample_data(cam_token)获取标注数据
        'token': 记录当前传感器关键帧标注数据对应的token
        'sample_token'
        'ego_pose_token'
        'calibrated_sensor_token'
        'timestamp': 关键帧时间戳
        'fileformat'
        'is_key_frame'
        'height'
        'width'
        'filename'
        'prev'
        'next'
        'sensor_modality'
        'channel'
    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)  # 这里和scene内容保持一致
        scene = {
            "token": scene_rec["token"],
            "nbr_samples": scene_rec["nbr_samples"],
            "first_sample_token": scene_rec["first_sample_token"],
            "last_sample_token": scene_rec["last_sample_token"],
            "name": scene_rec["name"],
            "description": scene_rec["description"]
        }
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """


    from dataset.utils.custom_nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from dataset.utils.custom_nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.bevod_train #保持和train_val_split.json一致。
        val_scenes = splits.bevod_val     #保持和train_val_split.json一致。
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.根据lidartop的数据路径来判断
    available_scenes = get_available_scenes(nusc)  # list
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )

    # 解析nuscenes原始数据
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, list(train_scenes), list(val_scenes), test, max_sweeps=max_sweeps
    )

    # metadata = dict(version=version)
    if test:
        total_test_sample = 0
        for k, v in train_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "test", k + ".json")
            total_test_sample += len(v)
            # 序列化时处理 NumPy 数组
            v_json = json.loads(json.dumps(v, default=convert_numpy))
            with open(cache_path, "w", encoding='utf-8') as f:
                json.dump(v_json, open(cache_path, "w"), indent=True, ensure_ascii=False)
            print(
                "test sample: {}, val sample: {}".format(
                    total_train_sample, total_val_sample
                )
            )
    else:
        total_train_sample = 0
        for k, v in train_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "train", k + ".json")
            total_train_sample += len(v)
            # 序列化时处理 NumPy 数组
            v_json = json.loads(json.dumps(v, default=convert_numpy))
            with open(cache_path, "w", encoding='utf-8') as f:

                json.dump(v_json, open(cache_path, "w"), indent=True, ensure_ascii=False)

        total_val_sample = 0
        for k, v in val_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "val", k + ".json")
            total_val_sample += len(v)
            # 序列化时处理 NumPy 数组
            v_json = json.loads(json.dumps(v, default=convert_numpy))
            with open(cache_path, "w", encoding='utf-8') as f:
                json.dump(v_json, open(cache_path, "w"), indent=True, ensure_ascii=False)

        print(
            "train sample: {}, val sample: {}".format(
                total_train_sample, total_val_sample
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument("--root_path", type=str, default="./data/nuscenes_format")
    parser.add_argument("--fisheye_data_path", type=str, default="./data/fisheye_data_2w_frame")
    parser.add_argument("--info_prefix", type=str, default="./data/BEVOD_nusc_anno_dumpjson_mini")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--max_sweeps", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.info_prefix + "/train", exist_ok=True)
    os.makedirs(args.info_prefix + "/val", exist_ok=True)
    os.makedirs(args.info_prefix + "/test", exist_ok=True)

    versions = args.version.split(",")
    for version in versions:
        create_nuscenes_infos(
            args.root_path,
            args.info_prefix,
            version=version,
            max_sweeps=args.max_sweeps,
        )

   