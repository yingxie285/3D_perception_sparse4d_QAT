import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import pickle

# from mmdet3d.core.bbox.box_np_ops import points_cam2img
# from mmdet3d.datasets import NuScenesDataset

import os.path as osp
import argparse
import json
import random
import string
from tqdm import tqdm
import uuid
from scipy.linalg import polar


def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R


from math import pi

class Box3D():
    def __init__(self):
        self.center = None
        self.wlh = None
        self.orientation_yaw_pitch_roll = None
        self.name = None
        self.token = None
        self.instance_token = None
        self.track_id = None
        self.prev_token = None
        self.next_token = None
        self.timestamp = None
        self.visibility = None
        self.gt_velocity = None
        self.prev = None
        self.next = None


def get_cam_intr(camera_internal):

    intr = np.array([
            [camera_internal["fx"], 0, camera_internal["cx"]],
            [0, camera_internal["fy"], camera_internal["cy"]],
            [0, 0, 1]
        ])

    return intr

def get_cam_distort(distortion):

    distort = np.array([
            distortion["k1"],
            distortion["k2"],
            distortion["k3"],
            distortion["k4"]
        ])

    return distort


def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1).reshape(3, 1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2).reshape(3, 1)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation).reshape(3)

    return rotation, translation


visibility_mappings = {
    0: 4,
    1: 3,
    2: 2,
    3: 1
}


class_names_parking_mappings = {
    # 动态车辆类
    '普通轿车': 'car',
    '敞篷轿车': 'car',
    'SUV': 'car',
    'MPV': 'car',
    '面包车': 'car',
    '皮卡': 'car',
    '警车': 'car',
    '救护车': 'car',
    '房车': 'truck',
    '普通小型货车': 'truck',
    '箱式小型货车': 'truck',
    '普通大型货车': 'truck',
    '轿运车': 'truck',
    '客车': 'bus',
    '校车': 'bus',
    '普通公交车': 'bus',
    '铰链公交车': 'bus',
    '有轨电车': 'bus',
    '无轨电车': 'bus',
    # 特种/施工车辆 → 工程车辆
    '消防车': 'construction_vehicle',
    '清洁车': 'construction_vehicle',
    '工程车': 'construction_vehicle',
    '拖车': 'construction_vehicle',
    '拖拉机': 'construction_vehicle',
    '叉车': 'construction_vehicle',
    '油罐车': 'construction_vehicle',
    # 行人类
    '成人': 'pedestrian',
    '儿童': 'pedestrian',
    '交警': 'pedestrian',
    '环卫工人': 'pedestrian',
    '道路施工人员': 'pedestrian',
    # 摩托/慢速交通工具
    '两轮摩托': 'motor',
    '三轮摩托': 'motor',
    '两轮电动车': 'bicycle',
    '三轮电动车': 'bicycle',
    '两轮自行车': 'bicycle',
    '三轮自行车': 'bicycle',
    '滑板车': 'bicycle',
    '平衡车': 'bicycle',
    '婴儿车': 'bicycle',
    '轮椅': 'bicycle',
    '平板小推车': 'bicycle',
    '超市购物车': 'bicycle',
    '手推车': 'bicycle',
    '其他非机动车': 'bicycle',
    '未知非机动车': 'bicycle',
    '非机动车组': 'bicycle',
    # 动物
    '小型动物': 'animal',
    '大型动物': 'animal',
    # 静态结构：锥桶/挡板类
    '锥桶': 'traffic_cone',
    '防撞桶': 'barrier',
    '路桩': 'barrier',
    '石墩': 'barrier',
    '水马': 'barrier',
    # 静态结构：挡车器类
    '车位停止器': 'stopper',
    '车位锁': 'stopper',
    '减速带': 'stopper',
    # 静态结构：可移动杂物
    '垃圾桶': 'trash_bin',
    '灭火器': 'trash_bin',
    '箱子': 'trash_bin',
    # 静态标识类
    'A字牌': 'sign',
    '三角牌': 'sign',
    '施工警示牌': 'sign',
    '道闸杆': 'sign',
    'construcion_sign':'sign',
    # 其他/未分类障碍
    '柱子': 'barrier',
    '石块': 'barrier',
    '树枝/树杈': 'barrier',
    '空中漂浮物': 'barrier',
    '路坑/水洼': 'barrier',
    '其他静态障碍物': 'barrier',
    '其他机动车': 'car',
    '未知机动车': 'car',
    '未知机动车车轮': 'car',
    '未知机动车车灯': 'car',
}


def get_lidar_ego_global_infos(root_path,data_infos,v2x_side):
    lidar_ego_global_infos = {}
    for data_info in tqdm(data_infos):
        ## Step 1: build basic information
        sample_token = data_info['frame_id']
        lidar_ego_global_infos[sample_token] = {}

        if v2x_side == 'infrastructure-side':
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array([0, 0, 0]).reshape(3)

            calib_virtuallidar2global_path = osp.join(root_path, data_info['calib_virtuallidar_to_world_path'])  #虚拟激光雷达到全局坐标系的校准文件
            calib_virtuallidar2global = load_json(calib_virtuallidar2global_path)

            virtuallidar2word_rotation = np.array(calib_virtuallidar2global['rotation'])  #从 calib_virtuallidar_to_world_path 获取旋转矩阵
            approx_rotation_matrix = iterative_closest_point(virtuallidar2word_rotation)  #通过 iterative_closest_point 优化

            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=approx_rotation_matrix)))
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_virtuallidar2global['translation']).reshape(3)
        else:  # 车端
            calib_lidar2ego_path = osp.join(root_path, data_info['calib_lidar_to_novatel_path']) #ego2global_translation从车辆的 Novatel IMU 到全局坐标系的校准文件获取
            calib_lidar2ego = load_json(calib_lidar2ego_path)
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_lidar2ego['transform']['rotation']))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array(calib_lidar2ego['transform']['translation']).reshape(3)

            calib_ego2global_path = osp.join(root_path, data_info['calib_novatel_to_world_path'])
            calib_ego2global = load_json(calib_ego2global_path)
            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_ego2global['rotation']))))  #ego2global_rotation同样从 calib_novatel_to_world_path 获取
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_ego2global['translation']).reshape(3)
    
    return lidar_ego_global_infos

def cal_ego_velocity(data_infos,sample_info_mappings,lidar_ego_global_infos):
    #{'sample_token': [vx,xy]}
    ego_velocity = {}
    for data_info in tqdm(data_infos):
        sample_token = data_info['frame_id']
        cur_loc = lidar_ego_global_infos[sample_token]['ego2global_translation']
        cur_timestamp = float(sample_info_mappings[sample_token]['timestamp']) / 1e6

        next_sample_token = sample_info_mappings[sample_token]['next']
        if next_sample_token != '':
            next_loc = lidar_ego_global_infos[next_sample_token]['ego2global_translation']
            next_timestamp = float(sample_info_mappings[next_sample_token]['timestamp']) / 1e6
            ego_velocity[sample_token] = (next_loc - cur_loc) / (next_timestamp - cur_timestamp)
        else:
            ego_velocity[sample_token] = [0,0]
    
    return ego_velocity


def create_fisheye_infos(root_path,
                     out_path,
                     split_path,
                     can_bus_root_path,
                     info_prefix,
                     version='v1.0-mini',
                     max_sweeps=10,
                     split_part='train',
                     flag_save=True):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'cd
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    ## Step 0: load neccesary data
    data_info_path = osp.join(root_path, 'data_info.json')
    split_data_path = split_path

    data_infos = load_json(data_info_path)
    split_data = load_json(split_data_path)
    train_scenes = split_data['train']
    val_scenes = split_data['val']

    train_fisheye_infos = []
    val_fisheye_infos = []
    fisheye_infos = []

    sample_infos, sample_info_mappings = _generate_sample_infos(data_infos)  #生成样本信息和映射
    total_annotations = _get_total_annotations(root_path, data_infos, sample_info_mappings) #获取所有样本的标注信息
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings) #创建实例（对象）的 token 映射
    # total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings)  #为标注添加速度和前后帧信息
    #gen_infos
    camera_types = [
                   'FISHEYE_CAM_FRONT',
                   'FISHEYE_CAM_RIGHT',
                   'FISHEYE_CAM_LEFT',
                   'FISHEYE_CAM_BACK'
                   ]
    for split in ['train', 'val']:
        for data_info in tqdm(data_infos[split], desc=f'Processing {split}'):
            sample_token = data_info['frame_id']
            sample_info = sample_info_mappings[split][sample_token]  

            info = {
                'token': sample_info['token'],
                'frame_idx': sample_info['frame_idx'],
                'scene_token': sample_info['scene_token'],
                'timestamp': sample_info['timestamp'],
                'prev': sample_info['prev'],
                'next': sample_info['next'],
            }

            ## Step 2: build sensor data infomation
            info['front_image_path'] = data_info['front_image_path']
            info['right_image_path'] = data_info['right_image_path']
            info['left_image_path'] = data_info['left_image_path']
            info['back_image_path'] = data_info['back_image_path']
            info['lidar_path'] = data_info['pointcloud_path']
            info['ego_pose_path'] = data_info['ego_pose_path']
            # info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
            # info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
            # info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
            # info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

            info['cams'] = {}
            for camera_type in camera_types:
                info['cams'][camera_type] = {}
            info['cams'][camera_types[0]]['data_path'] = data_info['front_image_path']
            info['cams'][camera_types[1]]['data_path'] = data_info['right_image_path']
            info['cams'][camera_types[2]]['data_path'] = data_info['left_image_path']
            info['cams'][camera_types[3]]['data_path'] = data_info['back_image_path']

            key_calib_lidar2cam = 'cam_cali_path'              

            calib_lidar2cam_path = osp.join(root_path, data_info[key_calib_lidar2cam])
            calib_lidar2cam = load_json(calib_lidar2cam_path)
            for idx, camera_type in enumerate(camera_types):
                camera_internal = calib_lidar2cam[idx]['camera_internal']
                distortion = calib_lidar2cam[idx]['distortion']
                info['cams'][camera_type]['cam_intrinsic'] = get_cam_intr(camera_internal)
                info['cams'][camera_type]['distortion'] = get_cam_distort(distortion)
                info['cams'][camera_type]['camera_external'] = np.array(calib_lidar2cam[idx]['camera_external']).reshape(4, 4).T
                info['cams'][camera_type]['width'] = int(calib_lidar2cam[idx]['width'])
                info['cams'][camera_type]['height'] = int(calib_lidar2cam[idx]['height'])
            # info['cams'][camera_type]['lidar2cam_rotation'] = np.array(calib_lidar2cam['rotation'])
            # info['cams'][camera_type]['lidar2cam_translation'] = np.array(calib_lidar2cam['translation'])

            # cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
            # cam2lidar_t = - np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
            # info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
            # info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(3)

            # cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
            #                                 Quaternion(info['lidar2ego_rotation']).rotation_matrix,
            #                                 np.array(info['lidar2ego_translation']))
            # info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
            # info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)        

            # calib_cam_intrinsic_path = osp.join(root_path, data_info['calib_camera_intrinsic_path'])

            info['sweeps'] = {}
            info['can_bus'] = np.zeros(18)

            ## Step 3: build annotation information
            annotations = {}
                
            #检查 train 子字典
            if sample_token in total_annotations['train']:
                annotations.update(total_annotations['train'][sample_token])

            # 检查 val 子字典
            if sample_token in total_annotations['val']:
                annotations.update(total_annotations['val'][sample_token])

            boxes = []
            for anno_token in annotations.keys():
                annotation = annotations[anno_token]
                box3d = Box3D()
                box3d.center = [annotation['contour']['center3D']['x'], annotation['contour']['center3D']['y'],
                                annotation['contour']['center3D']['z']]
                box3d.wlh = [annotation['contour']['size3D']['y'], annotation['contour']['size3D']['x'],
                                annotation['contour']['size3D']['z']]
                box3d.orientation_yaw_pitch_roll = annotation['contour']['rotation3D']['z']
                box3d.name = annotation['type']
                # box3d.token = annotation['token']
                box3d.instance_token = annotation['instance_token']
                # box3d.track_id = int(annotation['trackId'])
                box3d.timestamp = float(sample_info['timestamp'])
                # box3d.visibility = visibility_mappings[annotation['occluded_state']]
                # box3d.gt_velocity = annotation['gt_velocity']
                # box3d.prev = annotation['prev']
                # box3d.next = annotation['next']

                boxes.append(box3d)

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation_yaw_pitch_roll
                                for b in boxes]).reshape(-1, 1)

            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)  #需要转换
            # gt_boxes = np.concatenate([locs, dims, -rots], axis=1)  #需要转换
            #图片Image0到Image3分别是前视、右视、左视、后视
            #从自车的前视摄像头来看，人位置在自车的右上角。从自车前进方向看，x轴正方向朝前，y轴正方向朝左。右手系
            #而从自车前进方向看，NuScenes的x轴正方指向右边，y轴正方向指向前方，z轴正方向垂直纸面向外。
            #所以，需要-rots - np.pi / 2来变换角度。
            names = np.array([b.name for b in boxes])
            instance_tokens = np.array([b.instance_token for b in boxes])
            # instance_inds = np.array([b.track_id for b in boxes])
            # box_tokens = np.array([b.token for b in boxes])
            timestamps = np.array([b.timestamp for b in boxes])
            # visibility_tokens = np.array([b.visibility for b in boxes])
            # gt_velocity = np.array([b.gt_velocity for b in boxes])
            # prev_anno_tokens = np.array([b.prev for b in boxes])
            # next_anno_tokens = np.array([b.next for b in boxes])

            # TODO: complete this part
            valid_flag = np.array([True for b in boxes])
            num_lidar_pts = np.array([1 for b in boxes])

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_ins_tokens'] = instance_tokens
            # info['gt_inds'] = instance_inds
            # info['anno_tokens'] = box_tokens
            # info['valid_flag'] = valid_flag
            # info['num_lidar_pts'] = num_lidar_pts
            info['timestamps'] = timestamps
            # info['visibility_tokens'] = visibility_tokens
            # info['gt_velocity'] = gt_velocity
            # info['prev_anno_tokens'] = prev_anno_tokens
            # info['next_anno_tokens'] = next_anno_tokens

            ## Step X: save fisheye infos
            if data_info['sequence_id'] in train_scenes:
                train_fisheye_infos.append(info)
            elif data_info['sequence_id'] in val_scenes:
                val_fisheye_infos.append(info)
            fisheye_infos.append(info)

            if flag_save:
                metadata = dict(version=version)
                data = dict(infos=train_fisheye_infos, metadata=metadata)
                info_path = osp.join(out_path,
                                        '{}_infos_temporal_train.pkl'.format(info_prefix))
                with open(info_path, 'wb') as f:
                    pickle.dump(data, f)

                data['infos'] = val_fisheye_infos
                info_val_path = osp.join(out_path,
                                        '{}_infos_temporal_val.pkl'.format(info_prefix))
                with open(info_val_path, 'wb') as f:
                    pickle.dump(data, f)


    return total_annotations, sample_info_mappings, fisheye_infos


def gen_token(*args):
    token_name = ''
    for value in args:
        token_name += str(value)
    token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
    return str(token)


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)

def get_single_sample_info(frame_id, data_infos):
    sample_info = {}
    for data in data_infos:
        if data['frame_id'] == frame_id:
            sample_info = data
            break
    return sample_info

def _generate_sample_infos(data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    sample_infos = {"train": [], "val": []}
    sample_info_mappings = {"train": {}, "val": {}}
    for split in ['train', 'val']:
        scene_data_dict = {}
        sample_mappings = {}
        for data_info in data_infos[split]:
            sample_token = data_info['frame_id']
            sample_mappings[sample_token] = data_info

            scene_token = data_info['sequence_id']
            if scene_token not in scene_data_dict.keys():
                scene_data_dict[scene_token] = []

            scene_data_dict[scene_token].append(sample_token)

        for scene_token in scene_data_dict.keys():
            scene_data_dict[scene_token].sort()

            for idx in range(len(scene_data_dict[scene_token])):
                info = {}
                if idx == 0:
                    info['prev'] = ''
                else:
                    info['prev'] = scene_data_dict[scene_token][idx - 1]
                
                if idx == len(scene_data_dict[scene_token]) - 1:
                    info['next'] = ''
                else:
                    info['next'] = scene_data_dict[scene_token][idx + 1]
                
                sample_token = scene_data_dict[scene_token][idx]
                sample_info = sample_mappings[sample_token]

                info['token'] = sample_info['frame_id']
                info['timestamp'] = float(sample_info['timestamp'])
                info['scene_token'] = sample_info['sequence_id']
                info['frame_idx'] = idx

                sample_infos[split].append(info)

        for sample_info in sample_infos[split]:
            sample_token = sample_info['token']
            sample_info_mappings[split][sample_token] = sample_info
    
    return sample_infos, sample_info_mappings

def _get_total_annotations_coop(root_path,data_infos,sample_info_mappings):
    total_annotations = {}
    for data_info in data_infos:
        sample_token = data_info['vehicle_frame']
        scene_token = sample_info_mappings[sample_token]['scene_token']

        annotation_path = osp.join(root_path, 'cooperative/label', sample_token+'.json')
        annotations = load_json(annotation_path)
        total_annotations[sample_token] = {}
        for annotation in annotations:
            anno_token = annotation['token']
            annotation["instance_token"] = gen_token(annotation['track_id'],scene_token)
            annotation['type'] = class_names_parking_mappings[annotation['type']]
            total_annotations[sample_token][anno_token] = annotation
    return total_annotations

def _get_total_annotations(root_path, data_infos, sample_info_mappings):
    total_annotations = {"train": {}, "val": {}}
    for split in ['train', 'val']:
        for data_info in data_infos[split]:
            sample_token = data_info['frame_id']
            scene_token = sample_info_mappings[split][sample_token]['scene_token']
            frame_idx = sample_info_mappings[split][sample_token]['frame_idx']
            timestamp = sample_info_mappings[split][sample_token]['timestamp']

            annotation_path = osp.join(root_path, data_info['anno_path'])
            annotations = load_json(annotation_path)
            total_annotations[split][sample_token] = {}
            for annotation in annotations['objects']:
                anno_token = annotation['id']
                annotation["instance_token"] = gen_token(annotation['trackName'], scene_token)
                annotation['type'] = class_names_parking_mappings[annotation['className']]
                # annotation["trackId"] = annotation['cube_index']
                total_annotations[split][sample_token][anno_token] = annotation
    return total_annotations


def _generate_unvisible_annotations(source_name, sample_info_mappings, secene_frame_mappings, instance_token_mappings,
                                    total_annotations,lidar_ego_global_infos):
    """Generate annotations for totally occluded objects and make trajectory complete.
    Args:
        root_path: data root
        data_infos: (list): data_infos loaded from data_info.json file.
    Return:
        dict[dict]: {'sample_token': {'anno_token': }}
    """
    ## Interpolate box   
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        cur_scene_token_end = cur_instance_samples[-1]['scene_token']
        assert cur_scene_token == cur_scene_token_end

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # linear interpolation
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii + 1]['annotation']['rotation']

                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[(cur_scene_token, cur_frame_idx)]
                # cur_time_stamp = sample_data_mappings[cur_sample_token]['pointcloud_timestamp']
                cur_timestamp = sample_info_mappings[cur_sample_token]['timestamp']

                # if cur_timestamp == '1626155888.384136':
                #     cur_timestamp = cur_timestamp

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']
                
                cur_loc = loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp,
                                                   lidar_ego_global_infos[sample_token_0],lidar_ego_global_infos[sample_token_1],
                                                   lidar_ego_global_infos[cur_sample_token])
                cur_rot = rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp)
                cur_anno_token = gen_token(source_name, cur_sample_token, str(cur_loc['x']), str(cur_loc['y']), str(cur_loc['z']))

                cur_instance_sample_anno = {
                    "token": cur_anno_token,
                    "type": cur_instance_samples[ii]['annotation']['type'],
                    "track_id": cur_instance_samples[ii]['annotation']['track_id'],
                    "truncated_state": 0,
                    "occluded_state": 3,
                    "3d_dimensions": cur_instance_samples[ii]['annotation']['3d_dimensions'],
                    "3d_location": cur_loc,
                    "rotation": cur_rot,
                    "instance_token": instance_token
                }

                total_annotations[cur_sample_token][cur_anno_token] = cur_instance_sample_anno

                cur_frame_idx = cur_frame_idx + 1

    return total_annotations


def loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp, \
                             lidar_ego_global_info_0,lidar_ego_global_info_1,cur_lidar_ego_global_info):
    """Use linear interpolation to estimate the 3d location for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    #cvt to global
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
    # lidar2ego
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['lidar2ego_rotation']).rotation_matrix, center_0)   \
                        + np.array(lidar_ego_global_info_0['lidar2ego_translation'])
    # ego2global
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['ego2global_rotation']).rotation_matrix, center_0)  \
                        + np.array(lidar_ego_global_info_0['ego2global_translation'])

    #cvt to global
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])
    # lidar2ego
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['lidar2ego_rotation']).rotation_matrix, center_1)   \
                        + np.array(lidar_ego_global_info_1['lidar2ego_translation'])
    # ego2global
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['ego2global_rotation']).rotation_matrix, center_1)  \
                        + np.array(lidar_ego_global_info_1['ego2global_translation'])
    
    # global interpolation
    cur_center = (center_1 - center_0) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + center_0

    # cur sesor data interpolation
    cur_ego2global_translation = cur_lidar_ego_global_info['ego2global_translation']
    cur_ego2global_rotation = Quaternion(cur_lidar_ego_global_info['ego2global_rotation'])

    # global to ego, ego to lidar
    global2ego_r = np.linalg.inv(cur_ego2global_rotation.rotation_matrix)
    global2ego_t = - np.array(cur_ego2global_translation).reshape(1, 3) @ global2ego_r.T
    global2ego_t = global2ego_t.reshape(3)

    ego2lidar_r = np.linalg.inv(Quaternion(cur_lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix)
    ego2lidar_t =  - np.array(cur_lidar_ego_global_info['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r.T
    ego2lidar_t = ego2lidar_t.reshape(3)

    # global2ego
    cur_center = np.dot(global2ego_r, cur_center) + global2ego_t 
    # ego2lidar
    cur_center = np.dot(ego2lidar_r, cur_center) + ego2lidar_t 
                       
    cur_loc = {}
    cur_loc['x'] = cur_center[0]
    cur_loc['y'] = cur_center[1]
    cur_loc['z'] = cur_center[2]

    return cur_loc


def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    time_ratio = (cur_timestamp - timestamp_ii_0) / (timestamp_ii_1 - timestamp_ii_0)
    diff = rot_ii_1 - rot_ii_0 + pi
    if diff < 0:
        diff = diff + pi
    elif diff > 2*pi:
        diff = diff -3*pi
    else:
        diff = diff - pi
    
    cur_rot = time_ratio * diff + rot_ii_0
    if cur_rot < -pi:
        cur_rot = cur_rot + 2*pi
    if cur_rot > 2*pi:
        cur_rot = cur_rot - 2*pi
    
    return cur_rot



# def add_velocity_simple(instance_token_mappings):
#     # 假设 instance_token_mappings['train'/'val'][instance_token] 是按时间排序的样本列表
#     for split in ['train','val']:
#         for inst_tok, samples in instance_token_mappings[split].items():
#             for i in range(len(samples)):
#                 # prev/next token
#                 prev_tok = '' if i==0 else samples[i-1]['annotation']['token']
#                 next_tok = '' if i==len(samples)-1 else samples[i+1]['annotation']['token']

#                 if i==len(samples)-1:
#                     gt_v = [0.0,0.0]
#                 else:
#                     c0 = np.array([samples[i]['annotation']['3d_location']['x'],
#                                    samples[i]['annotation']['3d_location']['y'],
#                                    samples[i]['annotation']['3d_location']['z']],dtype=float)
#                     c1 = np.array([samples[i+1]['annotation']['3d_location']['x'],
#                                    samples[i+1]['annotation']['3d_location']['y'],
#                                    samples[i+1]['annotation']['3d_location']['z']],dtype=float)
#                     t0 = float(samples[i]['timestamp'])/1e6
#                     t1 = float(samples[i+1]['timestamp'])/1e6
#                     dt = t1 - t0
#                     if dt<=0:
#                         gt_v = [0.0,0.0]
#                     else:
#                         v = (c1 - c0) / dt
#                         gt_v = [float(v[0]), float(v[1])]  # 只取平面速度x,y

#                 samples[i]['annotation']['gt_velocity'] = gt_v
#                 samples[i]['annotation']['prev'] = prev_tok
#                 samples[i]['annotation']['next'] = next_tok
#     return instance_token_mappings



# def _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings):
#     """Generate velocity and prev/next token for annotations.
#     Args:
#         total_annotations: added occluded annotations
#         sample_info_mappings
#         data_infos
#     """
#     ## Generate Velocity and Successors
#     for split in ['train', 'val']:
#         for instance_token in instance_token_mappings[split].keys():
#             cur_instance_samples = instance_token_mappings[split][instance_token]
#             cur_scene_token = cur_instance_samples[0]['scene_token']
            
#             for ii in range(len(cur_instance_samples)):
#                 if ii == 0:
#                     prev_anno_token = ''
#                 else:
#                     prev_anno_token = cur_instance_samples[ii - 1]['annotation']['token']

#                 if ii == len(cur_instance_samples) - 1:
#                     next_anno_token = ''
#                 else:
#                     next_anno_token = cur_instance_samples[ii + 1]['annotation']['token']

#                 if ii == len(cur_instance_samples) - 1:
#                     gt_velocity = [0, 0]
#                 else:
#                     loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
#                     loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']

#                     sample_token_0 = cur_instance_samples[ii]['sample_token']
#                     sample_token_1 = cur_instance_samples[ii+1]['sample_token']

#                     # cvt to global
#                     center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
#                     center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])

#                     # -------------------------计算速度-------------------------------------
#                     # lidar2ego
#                     # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix, center_0)   \
#                     #                     + np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation'])
                  

#                     center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['lidar2ego_rotation']).rotation_matrix, center_1)   \
#                                         + np.array(lidar_ego_global_infos[sample_token_1]['lidar2ego_translation'])

#                     # ego2global
#                     # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix, center_0)  \
#                     #                     + np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation'])
#                     center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['ego2global_rotation']).rotation_matrix, center_1)  \
#                                         + np.array(lidar_ego_global_infos[sample_token_1]['ego2global_translation']) 
                    
#                     global2ego_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix)
#                     global2ego_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation']).reshape(1, 3) @ global2ego_r0.T
#                     global2ego_t0 = global2ego_t0.reshape(3)

#                     ego2lidar_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix)
#                     ego2lidar_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r0.T
#                     ego2lidar_t0 = ego2lidar_t0.reshape(3)                

#                     # ego2lidar
#                     center_1 = np.dot(global2ego_r0,center_1) + global2ego_t0  
#                     center_1 = np.dot(ego2lidar_r0,center_1) + ego2lidar_t0                         

#                     # time_delta
#                     timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
#                     timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']
#                     timestamp_ii_0 = float(timestamp_ii_0) / 1e6
#                     timestamp_ii_1 = float(timestamp_ii_1) / 1e6
#                     time_delta = timestamp_ii_1 - timestamp_ii_0

#                     # gt_velocity_dict = {}
#                     # for key in loc_ii_0.keys():
#                     #     gt_velocity_dict[key] = (loc_ii_1[key] - loc_ii_0[key]) / (timestamp_ii_1 - timestamp_ii_0)
#                     # gt_velocity = [gt_velocity_dict['x'], gt_velocity_dict['y']]
#                     gt_velocity = (center_1 - center_0) / time_delta
#                     gt_velocity = gt_velocity[:2]
#                     # -------------------------计算速度-------------------------------------



#                 instance_token_mappings[instance_token][ii]['annotation']['gt_velocity'] = gt_velocity
#                 instance_token_mappings[instance_token][ii]['annotation']['prev'] = prev_anno_token
#                 instance_token_mappings[instance_token][ii]['annotation']['next'] = next_anno_token

#     return total_annotations, instance_token_mappings

def _get_secene_frame_mappings(sample_info_mappings):
    secene_frame_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        secene_frame_mappings[(scene_token, frame_idx)] = sample_token

    return secene_frame_mappings


def _get_instance_token_mappings(total_annotations, sample_info_mappings):
    instance_token_mappings = {"train": {}, "val": {}}

    for split in ['train', 'val']:
        for sample_token in total_annotations[split].keys():
            annotations = total_annotations[split][sample_token]
            scene_token = sample_info_mappings[split][sample_token]['scene_token']
            frame_idx = sample_info_mappings[split][sample_token]['frame_idx']
            timestamp = sample_info_mappings[split][sample_token]['timestamp']

            for anno_token in annotations.keys():
                annotation = annotations[anno_token]
                instance_token = annotation["instance_token"]
                if instance_token not in instance_token_mappings[split].keys():
                    instance_token_mappings[split][instance_token] = []
                instance_token_mappings[split][instance_token].append({
                    'scene_token': scene_token,
                    'frame_idx': frame_idx,
                    'sample_token': sample_token,
                    'timestamp': timestamp,
                    'annotation': annotation})

        # sorted by frame_idx, for downstream usage
        for instance_token in instance_token_mappings[split].keys():
            sorted(instance_token_mappings[split][instance_token], key=lambda annotation: annotation['frame_idx'])

    return instance_token_mappings


def generate_json_maps_files(data_root, version='v1.0-mini'):
    json_types = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                  'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

    import shutil
    if not os.path.exists(osp.join(data_root, version)):
        tmp_nuscenes_json_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/v1.0-mini'
        shutil.copytree(tmp_nuscenes_json_root, osp.join(data_root, version))

    if not os.path.exists(osp.join(data_root, 'maps')):
        tmp_nuscenes_map_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/maps'
        shutil.copytree(tmp_nuscenes_map_root, osp.join(data_root, 'maps'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./data/fisheye_data_2w_frame")
    parser.add_argument('--save-root', type=str, default="./data/nuscenes_anno_pkls")
    parser.add_argument('--split-file', type=str, default="./data/fisheye_data_2w_frame/train_val_split.json")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="fisheye")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    import os

    curDirectory = os.getcwd()
    basepath = os.path.basename(os.path.normpath(curDirectory))
    if basepath != 'Sparse4D_no_tracking':
        os.chdir('Sparse4D_no_tracking/')

    curDirectory = os.getcwd()
    print(curDirectory)

    data_root = args.data_root
    save_root = args.save_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix

    print(data_root)
    print(save_root)

    # generate_json_maps_files(data_root, version=v2x_side)
    total_annotations, sample_info_mappings, fisheye_infos = create_fisheye_infos(data_root,
                                                                        save_root,
                                                                        split_path,
                                                                        can_bus_root_path,
                                                                        info_prefix,
                                                                        version=args.version,
                                                                        max_sweeps=10)
    
