import argparse
import shutil
import os
import json
import os.path as osp
import uuid
import pyquaternion
import math
import numpy as np
from uuid import uuid4

from bevod_to_sparse import create_fisheye_infos, _get_instance_token_mappings
from bevod_to_sparse import load_json, write_json, visibility_mappings
from tqdm import tqdm

def generate_category_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating category json ---------------------")
    if not os.path.exists(osp.join(save_root, version)):
        os.makedirs(osp.join(save_root, version), exist_ok=True)

    sr_file = osp.join(local_root, 'category.json')
    #  加载并添加 token
    with open(sr_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)
        for item in categories:
            item["token"] = str(uuid.uuid4()).replace("-", "")  

    #  保存新的 category.json
    target_file_path = osp.join(save_root, version, 'category.json')
    with open(target_file_path, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=4, ensure_ascii=False)

    print(f"✔ category.json 已生成，共 {len(categories)} 类 → {target_file_path}")


def generate_attribute_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating attribute json ---------------------")
    if not os.path.exists(osp.join(save_root, version)):
        os.mkdir(osp.join(save_root, version))

    sr_file = osp.join(local_root, 'attribute.json')
    target_file_path = osp.join(save_root, version, 'attribute.json')
    shutil.copy(sr_file, target_file_path)


def generate_visibility_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating visibility json ---------------------")
    if not os.path.exists(osp.join(save_root, version)):
        os.mkdir(osp.join(save_root, version))

    sr_file = osp.join(local_root, 'visibility.json')
    target_file_path = osp.join(save_root, version, 'visibility.json')
    shutil.copy(sr_file, target_file_path)


def generate_instance_json(total_annotations, sample_info_mappings, save_root, version='v1.0-trainval'):
    print("--------------------- Start generating instance json ---------------------")
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)
    category_root = osp.join(save_root, version)
    category_data = load_json(osp.join(category_root, 'category.json'))
    category_token_mappings = {}
    for category_type in category_data:
        category_token_mappings[category_type['name']] = category_type['token']

    instance_json_datas = []  
    for split in ['train', 'val']:
        for instance_token in instance_token_mappings[split].keys():  #暂时跟踪不了
            cur_instance_samples = instance_token_mappings[split][instance_token] #instance_token_mappings中，这个实例在数据集中出现了多少帧，现在跟踪是错误的
            nbr_annotations = len(cur_instance_samples)
            category_name = cur_instance_samples[0]['annotation']['type']

            json_data = {
                'token': instance_token,
                'category_token': category_token_mappings[category_name],
                'nbr_annotations': nbr_annotations,
                # 'first_annotation_token': cur_instance_samples[0]['annotation']['token'],   #暂时跟踪不了
                # 'last_annotation_token': cur_instance_samples[nbr_annotations - 1]['annotation']['token']
            }

            instance_json_datas.append(json_data)
    

    target_file_path = osp.join(save_root, version, 'instance.json')
    write_json(instance_json_datas, target_file_path)


def generate_sensor_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating sensor json ---------------------")
    if not os.path.exists(osp.join(save_root, version)):
        os.mkdir(osp.join(save_root, version))

    sr_file = osp.join(local_root, 'sensor.json')
   
    target_file_path = osp.join(save_root, version, 'sensor.json')
    
    #直接复制，不修改 token，不生成 uuid
    shutil.copyfile(sr_file, target_file_path)

    print(f"✔ sensor.json 已复制 → {target_file_path}")

#0128 用每一帧每个相机独立的内外参
def generate_calibrated_sensor_json(fisheye_infos, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating calibrated_sensors(camere2lidar) json ---------------------")
    
    # 1. 加载硬件映射
    sensor_json_path = osp.join(save_root, version, 'sensor.json')
    sensor_data = load_json(sensor_json_path)
    sensor_channel_to_token = {s['channel']: s['token'] for s in sensor_data}

    # 列表顺序：相机在前，LIDAR 在后
    calibrated_sensors_list = ['FISHEYE_CAM_FRONT', 'FISHEYE_CAM_RIGHT', 'FISHEYE_CAM_LEFT', 'FISHEYE_CAM_BACK', 'LIDAR']
    calibrated_sensor_infos = []

    # 2. 遍历每一帧
    for frame in tqdm(fisheye_infos, desc="Generating calibrated sensors"):
        
        for channel in calibrated_sensors_list:
            # 为当前帧的这个传感器生成唯一 token
            calibrated_token = uuid4().hex
            
            # 1. 处理相机 (前四个)
            if channel.startswith('FISHEYE_CAM'):
                if channel not in frame['cams']:
                    continue
                
                cam_data = frame['cams'][channel]
                info_camera = {
                    'token': calibrated_token,
                    'sensor_token': sensor_channel_to_token[channel],
                    'camera_extrinsic': cam_data['camera_external'].tolist(),
                    'camera_intrinsic': cam_data['cam_intrinsic'].tolist(),
                    'distortion': cam_data['distortion'].tolist()
                }
                calibrated_sensor_infos.append(info_camera)
                cam_data['calibrated_sensor_token'] = calibrated_token

            # 2. 处理 LIDAR (最后一个)
            elif channel == 'LIDAR':
                info_lidar = {
                    'token': calibrated_token,
                    'sensor_token': sensor_channel_to_token[channel],
                    'camera_extrinsic': [],
                    'camera_intrinsic': [],
                    'distortion': [] 
                }
                calibrated_sensor_infos.append(info_lidar)
                
                # 【关键】回写给 LIDAR
                # 哪怕没有内外参，这个 token 也要存下来，否则 sample_data 找不到它
                if 'lidar' in frame:
                    frame['lidar']['calibrated_sensor_token'] = calibrated_token
                else:
                    # 如果 frame 下面没有单独的 lidar 字典，就直接挂在 frame 上
                    frame['lidar_calibrated_sensor_token'] = calibrated_token

    # 4. 保存
    target_file_path = osp.join(save_root, version, 'calibrated_sensor.json')
    write_json(calibrated_sensor_infos, target_file_path)
    print(f"Done! Saved {len(calibrated_sensor_infos)} entries to {target_file_path}")    



# def generate_calibrated_sensor_json(fisheye_infos, save_root, version='v1.0-trainval', local_root=''):
#     print("--------------------- Start generating calibrated_sensors(camere2lidar) json ---------------------")

#     sensor_json_path = osp.join(save_root, version)
#     sensor_data = load_json(osp.join(sensor_json_path, 'sensor.json'))
#     calibrated_sensors = ['FISHEYE_CAM_FRONT', 'FISHEYE_CAM_RIGHT','FISHEYE_CAM_LEFT','FISHEYE_CAM_BACK','LIDAR']

#     calibrated_sensor_infos = []
#     for sensor_info in sensor_data:
#         print(sensor_info['channel'])
#         if sensor_info['channel'] in calibrated_sensors:
#             if sensor_info['modality'] == 'camera':
#                 camera_extrinsic = fisheye_infos[0]['cams'][sensor_info['channel']]['camera_external']
#                 camera_intrinsic = fisheye_infos[0]['cams'][sensor_info['channel']]['cam_intrinsic']
#                 token = uuid4().hex
#                 distortion = fisheye_infos[0]['cams'][sensor_info['channel']]['distortion']
#                 info_camera = {
#                     'token': token,
#                     'sensor_token': sensor_info['token'],
#                     'camera_extrinsic': camera_extrinsic.tolist(),
#                     'camera_intrinsic': camera_intrinsic.tolist(),
#                     'distortion':distortion.tolist()
#                 }
#                 calibrated_sensor_infos.append(info_camera)
#             else:
#                 token = uuid4().hex
#                 info_lidar = {
#                     'token': token,
#                     'sensor_token': sensor_info['token'],
#                     'camera_extrinsic': [],
#                     'camera_intrinsic': [],
#                 }
#                 calibrated_sensor_infos.append(info_lidar)

#     target_file_path = osp.join(save_root, version, 'calibrated_sensor.json')
#     write_json(calibrated_sensor_infos, target_file_path)

def generate_ego_pose_json(fisheye_infos, data_root, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating ego_pose json ---------------------")
    #提取欧拉角
    # JSON文件路径
    base_dir = osp.join(data_root)
    result = []

    token2trans = {}  # 新增：token -> translation 映射
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "position":
            # 排序：按文件名（去掉后缀）转int后升序
            files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    token = data.get("name")
                    pos = data.get("position", {})
                    roll = pos.get("roll")
                    pitch = pos.get("pitch")
                    yaw = pos.get("yaw")
                    x = pos.get("x")
                    y = pos.get("y")
                    z = pos.get("z")
                    result.append({
                        "token": token,
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                        "x": x,
                        "y": y,
                        "z": z
                    })
                    token2trans[token] = [x, y, z]  #  加入 translation 映射

    quaternion_results = []
    token2quat = {}  # 新增：token到quaternion的映射

    for item in result:
        token = item['token']
        euler_angles = [item['roll'], item['pitch'], item['yaw']]
        quat = EulerAndQuaternionTransform(euler_angles, angle_is_not_rad=False)  # 这里的角度是弧度，传False
        quaternion_results.append({
            'token': token,
            'quaternion': quat
        })
        token2quat[token] = quat  # 映射添加

    ego_pose_infos = []
    for  fisheye_info in fisheye_infos:
        # token_lidar = lidar_tokens[i]  # 每帧的 token 保持一致
        info = {
            'token': fisheye_info['token'],
            'timestamp': fisheye_info['timestamp'],
            'rotation': token2quat[fisheye_info['token']],
            'translation':token2trans[fisheye_info['token']]
        }
        ego_pose_infos.append(info)
    
    target_file_path = osp.join(save_root, version, 'ego_pose.json')
    write_json(ego_pose_infos, target_file_path)

    print(f" ego_pose.json已保存至：{target_file_path}")
   

def EulerAndQuaternionTransform(input_data, angle_is_not_rad=False):
    """
    四元数与欧拉角互换(Z-Y-X顺序)
    输入:
        input_data: 欧拉角[r,p,y]或四元数[w,x,y,z]
        angle_is_not_rad: 若为True,输入/输出角度为度数；否则为弧度
    输出:
        转换后的四元数或欧拉角（带逗号的列表形式，保留原始精度）
    """
    if len(input_data) == 3:
        # 欧拉角转四元数
        r, p, y = input_data
        if angle_is_not_rad:
            r = math.radians(r)
            p = math.radians(p)
            y = math.radians(y)

        cr = math.cos(r/2)
        cp = math.cos(p/2)
        cy = math.cos(y/2)
        sr = math.sin(r/2)
        sp = math.sin(p/2)
        sy = math.sin(y/2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        # 返回原始精度的列表
        return [w, x, y, z]
    elif len(input_data) == 4:
        # 四元数转欧拉角
        w, x, y, z = input_data
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(min(max(2 * (w * y - z * x), -1.0), 1.0))  # 限制asin输入范围[-1,1]
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        if angle_is_not_rad:
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
            
        # 返回原始精度的列表
        return [r, p, y]
    else:
        raise ValueError("输入数据长度必须为3(欧拉角)或4(四元数)")

def generate_log_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating log json ---------------------")

    if not os.path.exists(osp.join(save_root, version)):
        os.mkdir(osp.join(save_root, version))

    sr_file = osp.join(local_root, 'log.json')
    target_file_path = osp.join(save_root, version, 'log.json')
    shutil.copy(sr_file, target_file_path)


def generate_scene_json(sample_info_mappings, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating scene json ---------------------")

    scene_mappings = {}
    for split in ['train', 'val']:
        for sample_token in sample_info_mappings[split]:
            scene_token = sample_info_mappings[split][sample_token]['scene_token']
            if scene_token not in scene_mappings.keys():
                scene_mappings[scene_token] = []

            scene_mappings[scene_token].append(sample_info_mappings[split][sample_token])

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    scene_infos = []
    for scene_token in scene_mappings.keys():
        nbr_samples = len(scene_mappings[scene_token])
        # location = scene_mappings[scene_token][0]['location']
        info = {
            'token': scene_token,
            # 'log_token': log_mappings[location], #我们的sample_info_mappings[split][sample_token]没有'location'的键
            'nbr_samples': nbr_samples,
            'first_sample_token': scene_mappings[scene_token][0]['token'],
            'last_sample_token': scene_mappings[scene_token][nbr_samples - 1]['token'],
            'name': scene_token,
            'description': ''
        }

        scene_infos.append(info)

    target_file_path = osp.join(save_root, version, 'scene.json')
    write_json(scene_infos, target_file_path)



def generate_sample_json(sample_info_mappings, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating sample json ---------------------")

    def gen_token(sensor_name, sample_token):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_name = sensor_name + sample_token
        token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
        return str(token)

    sensor_name = 'FISHEYE_CAM_FRONT'
    for split in ['train', 'val']:
        for sample_token in sample_info_mappings[split].keys():
            sample_info_mappings[split][sample_token]['image_token'] = gen_token(sensor_name, sample_token)
    
    image_sample_infos = []
    for split in ['train', 'val']:
        for sample_token in sample_info_mappings[split].keys():
            sample_info = sample_info_mappings[split][sample_token]
            prev_token = '' if sample_info['prev'] == '' else sample_info_mappings[split][sample_info['prev']]['image_token']
            next_token = '' if sample_info['next'] == '' else sample_info_mappings[split][sample_info['next']]['image_token']
            info = {
                'token': sample_token,
                'timestamp': sample_info['timestamp'],
                'prev': sample_info['prev'],
                'next': sample_info['next'],
                'scene_token': sample_info['scene_token']
            }

            image_sample_infos.append(info)

    target_file_path = osp.join(save_root, version, 'sample.json')
    write_json(image_sample_infos, target_file_path)

#0128 用每一帧每个相机独立的内外参
def generate_sample_data_json(fisheye_infos, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating sample data json ---------------------")
    def gen_token(sensor_names, sample_token):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_list = []
        for sensor_name in sensor_names:
            token_name = sensor_name + sample_token
            token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
            token_list.append(str(token))
        return token_list
    
    sample_data_infos = []
    sensor_names = [
        'FISHEYE_CAM_FRONT',
        'FISHEYE_CAM_RIGHT',
        'FISHEYE_CAM_LEFT',
        'FISHEYE_CAM_BACK',
    ]
    for sample_info in fisheye_infos:
        token_list = gen_token(sensor_names, sample_info['token'])
        timestamp = sample_info['timestamp']

        cams = sample_info['cams']
        lidar_calibrated_sensor_token = sample_info['lidar_calibrated_sensor_token']

        info_front_image = {
            'token': token_list[0],
            'sample_token': sample_info['token'],
            'ego_pose_token': sample_info['token'],
            'calibrated_sensor_token': cams['FISHEYE_CAM_FRONT']['calibrated_sensor_token'],
            'timestamp': timestamp,
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': 1200,
            'width': 1920,
            'filename': sample_info['front_image_path'],
            'prev': sample_info['prev'],
            'next': sample_info['next']
        }
        sample_data_infos.append(info_front_image)

        info_right_image = {
            'token': token_list[1],
            'sample_token': sample_info['token'],
            'ego_pose_token': sample_info['token'],
            'calibrated_sensor_token': cams['FISHEYE_CAM_RIGHT']['calibrated_sensor_token'],
            'timestamp': timestamp,
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': 1200,
            'width': 1920,
            'filename': sample_info['right_image_path'],
            'prev': sample_info['prev'],
            'next': sample_info['next']
        }
        sample_data_infos.append(info_right_image)

        info_left_image = {
            'token': token_list[2],
            'sample_token': sample_info['token'],
            'ego_pose_token': sample_info['token'],
            'calibrated_sensor_token': cams['FISHEYE_CAM_LEFT']['calibrated_sensor_token'],
            'timestamp': timestamp,
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': 1536,
            'width': 1920,
            'filename': sample_info['left_image_path'],
            'prev': sample_info['prev'],
            'next': sample_info['next']
        }
        sample_data_infos.append(info_left_image)

        info_back_image = {
            'token': token_list[3],
            'sample_token': sample_info['token'],
            'ego_pose_token': sample_info['token'],
            'calibrated_sensor_token': cams['FISHEYE_CAM_BACK']['calibrated_sensor_token'],
            'timestamp': timestamp,
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': 1200,
            'width': 1920,
            'filename': sample_info['back_image_path'],
            'prev': sample_info['prev'],
            'next': sample_info['next']
        }
        sample_data_infos.append(info_back_image)
        info_lidar = {
                # 'token_lidar':token_lidar,
                'token': sample_info['token'],
                'sample_token': sample_info['token'],
                'ego_pose_token': sample_info['token'],
                'calibrated_sensor_token': lidar_calibrated_sensor_token,
                'timestamp': sample_info['timestamp'],
                'fileformat': 'npy',   #1216
                'is_key_frame': bool(1),
                'height': 0,
                'width': 0,
                'filename': sample_info['lidar_path'],
                'prev': sample_info['prev'],
                'next': sample_info['next']
        }
        sample_data_infos.append(info_lidar)
        
    target_file_path = osp.join(save_root, version, 'sample_data.json')
    write_json(sample_data_infos, target_file_path)


# def generate_sample_data_json(fisheye_infos, save_root, version='v1.0-trainval', local_root=''):
#     print("--------------------- Start generating sample data json ---------------------")

#     calibrated_sensor_data = load_json(osp.join(save_root, version, 'calibrated_sensor.json')) #前右左后lidar
#     calibrated_sensor_token0 = calibrated_sensor_data[0]['token']
#     calibrated_sensor_token1 = calibrated_sensor_data[1]['token']
#     calibrated_sensor_token2 = calibrated_sensor_data[2]['token']
#     calibrated_sensor_token3 = calibrated_sensor_data[3]['token']
#     calibrated_sensor_token4 = calibrated_sensor_data[4]['token']

#     sample_data_infos = []
#     def gen_token(sensor_names, sample_token):
#         """
#             Args:
#                 sensor_name: "VEHICLE_CAM_FRONT"
#                 str_x: str label["3d_location"]["x"]
#                 str_y: str label["3d_location"]["y"]
#                 str_z: str label["3d_location"]["z"]
#             Returns:
#                 str(token)
#         """
#         token_list = []
#         for sensor_name in sensor_names:
#             token_name = sensor_name + sample_token
#             token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
#             token_list.append(str(token))
#         return token_list
    
#     sensor_names = [
#         'FISHEYE_CAM_FRONT',
#         'FISHEYE_CAM_RIGHT',
#         'FISHEYE_CAM_LEFT',
#         'FISHEYE_CAM_BACK',
#     ]

#     for sample_info in fisheye_infos: 
#             token_list = gen_token(sensor_names, sample_info['token'])
#             info_front_image = {
#                 # 'token_front_image':token_front_image,
#                 'token': token_list[0],
#                 'sample_token': sample_info['token'],
#                 'ego_pose_token': sample_info['token'],
#                 'calibrated_sensor_token': calibrated_sensor_token0,
#                 'timestamp': sample_info['timestamp'],
#                 'fileformat': 'jpg',
#                 'is_key_frame': bool(1),
#                 'height': 1200,
#                 'width': 1920,
#                 'filename': sample_info['front_image_path'],
#                 'prev': sample_info['prev'],
#                 'next': sample_info['next']
#             }
#             sample_data_infos.append(info_front_image)
#             info_right_image = {
#                 # 'token_right_image':token_right_image,
#                 'token': token_list[1],
#                 'sample_token': sample_info['token'],
#                 'ego_pose_token': sample_info['token'],
#                 'calibrated_sensor_token': calibrated_sensor_token1,
#                 'timestamp': sample_info['timestamp'],
#                 'fileformat': 'jpg',
#                 'is_key_frame': bool(1),
#                 'height': 1200,
#                 'width': 1920,
#                 'filename': sample_info['right_image_path'],
#                 'prev': sample_info['prev'],
#                 'next': sample_info['next']
#             }
#             sample_data_infos.append(info_right_image)
#             info_left_image = {
#                 # 'token_left_image':token_left_image,
#                 'token': token_list[2],
#                 'sample_token': sample_info['token'],
#                 'ego_pose_token': sample_info['token'],
#                 'calibrated_sensor_token': calibrated_sensor_token2,
#                 'timestamp': sample_info['timestamp'],
#                 'fileformat': 'jpg',
#                 'is_key_frame': bool(1),
#                 'height': 1200,
#                 'width': 1920,
#                 'filename': sample_info['left_image_path'],
#                 'prev': sample_info['prev'],
#                 'next': sample_info['next']
#             }
#             sample_data_infos.append(info_left_image)
#             info_back_image = {
#                 # 'token_back_image':token_back_image,
#                 'token': token_list[3],
#                 'sample_token': sample_info['token'],
#                 'ego_pose_token': sample_info['token'],
#                 'calibrated_sensor_token': calibrated_sensor_token3,
#                 'timestamp': sample_info['timestamp'],
#                 'fileformat': 'jpg',
#                 'is_key_frame': bool(1),
#                 'height': 1200,
#                 'width': 1920,
#                 'filename': sample_info['back_image_path'],
#                 'prev': sample_info['prev'],
#                 'next': sample_info['next']
#             }
#             sample_data_infos.append(info_back_image)
#             info_lidar = {
#                 # 'token_lidar':token_lidar,
#                 'token': sample_info['token'],
#                 'sample_token': sample_info['token'],
#                 'ego_pose_token': sample_info['token'],
#                 'calibrated_sensor_token': calibrated_sensor_token4,
#                 'timestamp': sample_info['timestamp'],
#                 'fileformat': 'npy',   #1216
#                 'is_key_frame': bool(1),
#                 'height': 0,
#                 'width': 0,
#                 'filename': sample_info['lidar_path'],
#                 'prev': sample_info['prev'],
#                 'next': sample_info['next']
#             }
#             sample_data_infos.append(info_lidar)
        

#     target_file_path = osp.join(save_root, version, 'sample_data.json')
#     write_json(sample_data_infos, target_file_path)


def generate_sample_annotation_json(total_annotations, save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating sample annotation json ---------------------")
    import numpy as np
    from pyquaternion import Quaternion

    sample_annotation_infos = []
    for split in ['train', 'val']:
        for sample_token in total_annotations[split].keys():
            annotations = total_annotations[split][sample_token]

            # for info in fisheye_infos:
            #     if info['token'] == sample_token:
            #         sample_lidar2ego_rotation = Quaternion(info['lidar2ego_rotation'])
            #         sample_lidar2ego_translation = np.array(info['lidar2ego_translation'])
            #         sample_ego2global_rotation = Quaternion(info['ego2global_rotation'])
            #         sample_ego2global_translation = np.array(info['ego2global_translation'])
            #         break

            for anno_token in annotations.keys():
                annotation = annotations[anno_token]

                # cvt global
                center = np.array([annotation['contour']['center3D']['x'], annotation['contour']['center3D']['y'],
                                annotation['contour']['center3D']['z']])
                rot = Quaternion(axis=[0, 0, 1], radians=annotation['contour']['rotation3D']['z']) #univ2x的方法，转成四元数
                
                # # lidar2ego
                # center = np.dot(sample_lidar2ego_rotation.rotation_matrix, center) + sample_lidar2ego_translation
                # rot = sample_lidar2ego_rotation * rot

                # # ego2global
                # center = np.dot(sample_ego2global_rotation.rotation_matrix, center) + sample_ego2global_translation
                # rot = sample_ego2global_rotation * rot

                info = {
                    'token': annotation['id'],
                    'sample_token': sample_token,
                    'instance_token': annotation['instance_token'],
                    # 'visibility_token': visibility_mappings[annotation['occluded_state']],  #我们的数据集没标
                    'attribute_tokens': [],
                    'translation': center.tolist(),
                    'size': [annotation['contour']['size3D']['y'], annotation['contour']['size3D']['x'],
                            annotation['contour']['size3D']['z']],
                    'rotation': rot.elements.tolist(),
                    # 'prev': annotation['prev'],  #我们的数据集没标
                    # 'next': annotation['next'],  #我们的数据集没标
                    'num_lidar_pts': 100, #univ2x固定为100
                    'num_radar_pts': 0   #univ2x固定为0
                }

                sample_annotation_infos.append(info)

    target_file_path = osp.join(save_root, version, 'sample_annotation.json')
    write_json(sample_annotation_infos, target_file_path)


def generate_map_json(save_root, version='v1.0-trainval', local_root=''):
    print("--------------------- Start generating map json ---------------------")

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    map_infos = []
    for location in log_mappings.keys():
        info = {
            'category': 'semantic_prior',
            'token': location,
            'filename': '',
            'log_tokens': [log_mappings[location]]
        }

        map_infos.append(info)

    target_file_path = osp.join(save_root, version, 'map.json')
    write_json(map_infos, target_file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./data/BEVOD_mini")
    parser.add_argument('--save-root', type=str, default="./data/nuscenes_format")
    parser.add_argument('--split-file', type=str, default="./data/BEVOD_mini/train_val_split.json")
    parser.add_argument('--local-root', type=str, default="./tool/data_converter/nuscenes_jsons")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="fisheye")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    import os

    curDirectory = os.getcwd()
    basepath = os.path.basename(os.path.normpath(curDirectory))

    curDirectory = os.getcwd()
    print(curDirectory)

    version = args.version
    data_root = args.data_root
    save_root = args.save_root
        
    local_root = args.local_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix

    total_annotations, sample_info_mappings, fisheye_infos = create_fisheye_infos(data_root,
                                                                        save_root,
                                                                        split_path,
                                                                        can_bus_root_path,
                                                                        info_prefix,
                                                                        version=version,
                                                                        max_sweeps=10,
                                                                        flag_save=False)

 
    generate_category_json(save_root,
                           version=version,
                           local_root=local_root)
    
    generate_attribute_json(save_root,
                            version=version,
                            local_root=local_root)

    generate_visibility_json(save_root,
                             version=version,
                             local_root=local_root)

    generate_instance_json(total_annotations,
                           sample_info_mappings,
                           save_root,
                           version=version)

    generate_sensor_json(save_root,
                         version=version,
                         local_root=local_root)

    generate_calibrated_sensor_json(fisheye_infos,
                                    save_root,
                                    version=version,
                                    local_root=local_root)

    generate_log_json(save_root,
                      version=version,
                      local_root=local_root)

    generate_scene_json(sample_info_mappings,
                        save_root,
                        version=version,
                        local_root=local_root)

    generate_sample_json(sample_info_mappings,
                         save_root,
                         version=version,
                         local_root=local_root)


    generate_sample_data_json(fisheye_infos,
                              save_root,
                              version=version,
                              local_root=local_root)
        
    generate_ego_pose_json(fisheye_infos,
                           data_root,
                           save_root,
                           version=version,
                           local_root=local_root)

    generate_sample_annotation_json(total_annotations,
                                    save_root,
                                    version=version,
                                    local_root=local_root)

    generate_map_json(save_root,
                      version=version,
                      local_root=local_root)
