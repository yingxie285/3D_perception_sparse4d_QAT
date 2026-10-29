import numpy as np
import cv2
import json
import numpy as np
import os
from natsort import natsorted
import re
from pathlib import Path

def load_camera_params(sample_dir, image_path):
    """
    sample_dir: /.../20240112-145620_20240112-145636/
    返回 dict:
    cam_name -> {"K":.., "dist":.., "lidar2cam":.., "image_path":..}
    """
    with open(sample_dir, "r", encoding='utf-8') as f:
        cfg_list = json.load(f)
    cam_params = {}
    cam_name_mapping ={
        0: "front",
        1: "right",
        2: "left",
        3: "back",
    }
    for index, cam_cfg in enumerate(cfg_list):
        cam_name = cam_name_mapping[index]
        # K = np.array(cam_cfg["camera_internal"], dtype=np.float64)
        K = np.array([
                [cam_cfg["camera_internal"]["fx"], 0, cam_cfg["camera_internal"]["cx"]],
                [0, cam_cfg["camera_internal"]["fy"], cam_cfg["camera_internal"]["cy"]],
                [0, 0, 1]
            ], dtype=np.float64)
        dist = [cam_cfg["distortion"]['k1'],cam_cfg["distortion"]['k2'],cam_cfg["distortion"]['k3'],cam_cfg["distortion"]['k4']]
        lidar2cam = np.array(cam_cfg["camera_external"], dtype=np.float64).reshape(4, 4).T
        img_path = image_path[cam_name]
        cam_params[cam_name] = {
            "K": K,
            "dist": dist,
            "lidar2cam": lidar2cam,
            "img_path": img_path
        }
    return cam_params


def load_annotation(anno_path):
    """
    sample_dir: /.../20240112-145620_20240112-145636/
    返回 dict:
    cam_name -> {"K":.., "dist":.., "lidar2cam":.., "image_path":..}
    """
    with open(anno_path, "r", encoding='utf-8') as f:
        anno_list = json.load(f)
    annotation_list = anno_list['objects']
    annotations = []
    for index, annotation in enumerate(annotation_list):
        lwh = [annotation['contour']['size3D']['x'], annotation['contour']['size3D']['y'], annotation['contour']['size3D']['z']]
        center = [annotation['contour']['center3D']['x'], annotation['contour']['center3D']['y'], annotation['contour']['center3D']['z']]
        yaw = annotation['contour']['rotation3D']['z']
        annotations.append(
            {
              "lwh": lwh,
              "center": center,
              "yaw": yaw  
            }
        )
    return annotations

def get_3d_bbox_corners(center, size, yaw):
    l, w, h = size
    x_c, y_c, z_c = center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = R @ corners #乘yaw旋转矩阵
    corners = corners + np.array(center).reshape(3,1)
    return corners.T  # (8,3)

def project_points_fisheye(points_3d, K, dist, lidar2cam):
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0],1))]) # (8,3)变成(8,4)
    cam_points = (lidar2cam @ points_3d_h.T).T[:, :3]
    cam_points = cam_points.astype(np.float64)
    # 不删除点
    valid_mask = cam_points[:, 2] > 0

    K = np.array(K, dtype=np.float64)
    D = np.array(dist, dtype=np.float64)

    X = cam_points[:, 0]
    Y = cam_points[:, 1]
    Z = cam_points[:, 2]

    # 防止除零/后方点异常
    Z_safe = np.where(valid_mask, Z, 1e-6)

    r = np.sqrt(X ** 2 + Y ** 2)

    theta = np.arctan2(r, Z_safe)
    phi = np.arctan2(Y, X)

    k1, k2, k3, k4 = D.flatten()

    theta_d = theta * (
        1
        + k1 * theta ** 2
        + k2 * theta ** 4
        + k3 * theta ** 6
        + k4 * theta ** 8
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = cx + fx * theta_d * np.cos(phi)
    v = cy + fy * theta_d * np.sin(phi)

    img_points = np.stack([u, v], axis=1)
    
    return img_points, valid_mask

def draw_bbox(img, corners_2d,valid_mask, color=(0,255,0), thickness=2):
    """
    绘制 2D bbox 或投影后的 3D bbox。
    
    corners_2d: np.ndarray, shape (N,2)，投影后的角点坐标
    """
    if corners_2d is None or len(corners_2d) == 0:
        return img  # 没有点直接返回

    corners_2d = corners_2d.astype(int)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for i, j in edges:

        # 两个点都在相机前方才画
        if valid_mask[i] and valid_mask[j]:

            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[j])

            cv2.line(img, pt1, pt2, color, thickness)

    return img

def project_all_gt_to_cameras(cam_params, gt_labels, save_dir):
    """
    gt_labels: list of dict, 每个 dict 包含 center,size,yaw
    cam_params: dict, key=cam_name, value={"K":..,"dist":..,"lidar2cam":..,"image_path":..}
    save_dir: 保存可视化图片
    """
    for cam_name, params in cam_params.items():
        img = cv2.imread(params["img_path"])
        for gt in gt_labels:
            corners_3d = get_3d_bbox_corners(gt["center"], gt["lwh"], gt["yaw"]) # (8,3)
            corners_2d, valid_mask = project_points_fisheye(corners_3d, params["K"], params["dist"], params["lidar2cam"])
            if np.any(valid_mask):
                img = draw_bbox(img, corners_2d, valid_mask)
        os.makedirs(f'{save_dir}/{re.split(r"[/.]", params["img_path"])[-2]}', exist_ok=True)
        cv2.imwrite(f'{save_dir}/{re.split(r"[/.]", params["img_path"])[-2]}/{cam_name}_vis.jpg', img)

if __name__ == "__main__":
    dataset_dir = Path(os.environ["OCTPS_DATASET_DIR"]) / "data" / "fisheye_data_15w_frame"
    target_result_dir = Path(os.environ["TARGET_RESULT_DIR"]) / "Check_BEVOD_Label"

    scene_list = [
        d for d in natsorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, d))
    ][:3]
    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = os.path.join(target_result_dir, scene_dir.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)
        camera_config_dir = os.path.join(scene_dir, "camera_config")
        image_dir = {
            "front_dir": os.path.join(scene_dir, "image0"),
            "right_dir": os.path.join(scene_dir, "image1"),
            "left_dir": os.path.join(scene_dir, "image2"),
            "back_dir": os.path.join(scene_dir, "image3"),
        }
        annotation_dir = os.path.join(scene_dir, "result")
        cam_cfg_list = natsorted(os.listdir(camera_config_dir))
        anno_list    = natsorted(os.listdir(annotation_dir))
        front_list   = natsorted(os.listdir(image_dir["front_dir"]))
        right_list   = natsorted(os.listdir(image_dir["right_dir"]))
        left_list    = natsorted(os.listdir(image_dir["left_dir"]))
        back_list    = natsorted(os.listdir(image_dir["back_dir"]))
        for cam_cfg_json, anno_json, front_json, right_json, left_json, back_json in zip(cam_cfg_list, anno_list, front_list, 
                                                                                        right_list, left_list, back_list):
            cam_cfg_json_path = os.path.join(camera_config_dir, cam_cfg_json)
            anno_json_path = os.path.join(annotation_dir, anno_json)
            image_path ={
                        "front": os.path.join(image_dir["front_dir"], front_json),
                        "right": os.path.join(image_dir["right_dir"], right_json),
                        "left": os.path.join(image_dir["left_dir"], left_json),
                        "back": os.path.join(image_dir["back_dir"], back_json),
            }
            cam_params = load_camera_params(cam_cfg_json_path, image_path)
            annotations = load_annotation(anno_json_path)
            project_all_gt_to_cameras(cam_params, annotations, save_dir)