import numpy as np
import cv2
import json
import numpy as np
import os
from natsort import natsorted
import re


def load_camera_params(sample_dir, image_path):
    """
    sample_dir: /.../20240112-145620_20240112-145636/
    返回 dict:
    cam_name -> {"K":.., "dist":.., "lidar2cam":.., "image_path":..}
    """
    with open(sample_dir, "r") as f:
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
    with open(anno_path, "r") as f:
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
    mask = cam_points[:, 2] > 0
    cam_points = cam_points[mask].reshape(-1,1,3)
    if len(cam_points) == 0:
        return np.empty((0,2)) 
    K = np.array(K, dtype=np.float64)
    D = np.array(dist, dtype=np.float64)
    img_points, _ = cv2.fisheye.projectPoints(
        cam_points,
        rvec=np.zeros((3,1), dtype=np.float64),
        tvec=np.zeros((3,1), dtype=np.float64),
        K=K,
        D=D
    )
    return img_points.squeeze()  # (8,2)

def draw_bbox(img, corners_2d, color=(0,255,0), thickness=2):
    """
    绘制 2D bbox 或投影后的 3D bbox。
    
    corners_2d: np.ndarray, shape (N,2)，投影后的角点坐标
    """
    if corners_2d is None or len(corners_2d) == 0:
        return img  # 没有点直接返回

    corners_2d = corners_2d.astype(int)

    if corners_2d.ndim == 1 and corners_2d.shape[0] == 2:
        corners_2d = corners_2d[None, :]  # shape (1,2)
    
    n = len(corners_2d)

    if n == 1:
        # 单个点，画圆点
        pt = tuple(corners_2d[0])
        cv2.circle(img, pt, 2, color, thickness)
    elif n == 2:
        # 只有两点，画一条线
        pt1, pt2 = tuple(corners_2d[0]), tuple(corners_2d[1])
        cv2.line(img, pt1, pt2, color, thickness)
    elif n == 4:
        # 画矩形
        for i in range(4):
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4])
            cv2.line(img, pt1, pt2, color, thickness)
    elif n >= 8:
        # 原始 3D bbox 边
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i,j in edges:
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[j])
            cv2.line(img, pt1, pt2, color, thickness)
    else:
        # 其他点数，顺序连线
        for i in range(n-1):
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[i+1])
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
            corners_2d = project_points_fisheye(corners_3d, params["K"], params["dist"], params["lidar2cam"])
            if len(corners_2d) != 0:
                img = draw_bbox(img, corners_2d)
        os.makedirs(f'{save_dir}/{re.split(r"[/.]", params["img_path"])[-2]}', exist_ok=True)
        cv2.imwrite(f'{save_dir}/{re.split(r"[/.]", params["img_path"])[-2]}/{cam_name}_vis.jpg', img)

if __name__ == "__main__":
    dataset_dir = "/home/E2EModel/Model_Train_End/git-apa-perception-3D-OD/data/fisheye_data_15w_frame"
    scene_list = natsorted(os.listdir(dataset_dir))[:14]
    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = os.path.join("/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/Check_BEVOD_Label", scene_dir.split("/")[-1])
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