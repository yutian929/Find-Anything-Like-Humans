from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import DBSCAN


def convert_caminfo(caminfo_msg):
    """解析CameraInfo消息，返回4x4内参矩阵"""
    fx = caminfo_msg.K[0]
    fy = caminfo_msg.K[4]
    cx = caminfo_msg.K[2]
    cy = caminfo_msg.K[5]
    intrinsic = np.array(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return intrinsic


def convert_from_uvd(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    pose: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    将图像像素坐标 (u, v) 和深度图转换为世界坐标点云。

    Args:
        u (np.ndarray): 图像横向像素坐标 (N,)
        v (np.ndarray): 图像纵向像素坐标 (N,)
        depth (np.ndarray): 每个像素对应的深度值 (N,)，单位毫米
        intr (np.ndarray): 相机内参矩阵 (4x4)
        pose (np.ndarray): 相机在世界坐标系下的位姿变换矩阵 (4x4)
        depth_scale (float): 深度缩放因子，将深度值转换为米

    Returns:
        np.ndarray: 世界坐标系下的点云 (N, 3)
    """
    z = depth / depth_scale  # Convert depth to meters

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)

    uv = np.concatenate([u, v, padding], axis=0)  # Shape: (3, N)
    xyz = (np.linalg.inv(intr[:3, :3]) @ uv) * np.expand_dims(z, axis=0)
    xyz = np.concatenate([xyz, padding], axis=0)  # Homogeneous coords (4, N)
    xyz = pose @ xyz
    xyz[:3, :] /= xyz[3, :]  # Normalize homogeneous coords

    return xyz[:3, :].T  # Shape: (N, 3)


def convert_from_uvd_simu(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    image_width: int = 640,
    image_height: int = 480,
    hfov: float = 90.0,  # 水平视场角(度)
    depth_scale: float = 1.0,  # 仿真深度已经是米为单位，通常scale=1.0
) -> np.ndarray:
    """
    将仿真环境中的像素坐标 (u, v) 和深度值转换为世界坐标点云。
    针对仿真环境，深度值通常已经是以米为单位。

    Args:
        u (np.ndarray): 图像横向像素坐标 (N,)
        v (np.ndarray): 图像纵向像素坐标 (N,)
        depth (np.ndarray): 每个像素对应的深度值 (N,)，已经是米为单位
        pose (np.ndarray): 相机在世界坐标系下的位姿变换矩阵 (4x4)
        image_width (int): 图像宽度，像素
        image_height (int): 图像高度，像素
        hfov (float): 水平视场角，单位度
        depth_scale (float): 深度缩放因子，通常仿真环境为1.0

    Returns:
        np.ndarray: 世界坐标系下的点云 (N, 3)
    """
    # 将深度值转换为米（已经是米，所以只需要应用scale）
    z = depth / depth_scale

    # 计算焦距（以像素为单位）
    # 焦距 = (图像宽度/2) / tan(视场角/2)
    fx = (image_width / 2.0) / np.tan(np.radians(hfov / 2.0))
    fy = fx  # 假设像素纵横比为1:1

    # 图像中心点
    cx = image_width / 2.0
    cy = image_height / 2.0

    # 标准化u、v并转换为齐次坐标系
    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    z_expanded = np.expand_dims(z, axis=0)
    ones = np.ones_like(u)

    # 计算相机坐标系下的3D点
    x = (u - cx) * z_expanded / fx
    y = (v - cy) * z_expanded / fy

    # 组合为相机坐标系下的点云（齐次坐标）
    camera_points = np.vstack([x, y, z_expanded, ones])

    # 转换到世界坐标系
    world_points = pose @ camera_points

    # 处理齐次坐标
    world_points[:3, :] /= world_points[3, :]

    # 返回形状为(N, 3)的点云
    return world_points[:3, :].T


def random_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    从点云中随机采样若干点。

    Args:
        points (np.ndarray): 原始点云数据 (N, D)，N个点，每个点D维
        num_samples (int): 采样点数

    Returns:
        np.ndarray: 采样后的点云数据 (num_samples, D)
    """
    N = points.shape[0]
    if num_samples >= N:
        return points
    indices = np.random.choice(N, num_samples, replace=False)
    return points[indices]


def farthest_xyz_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    使用 Farthest Point Sampling (FPS) 方法从点云中采样。

    Args:
        points (np.ndarray): 原始点云数据 (N, 3)
        num_samples (int): 采样点数

    Returns:
        np.ndarray: 采样后的点云数据 (num_samples, 3)
    """
    N = points.shape[0]
    if num_samples >= N:
        return points

    sampled = np.zeros((num_samples,), dtype=np.int32)  # 存储采样点索引
    distances = np.ones((N,)) * 1e10  # 初始化每个点到已采样点的最小距离
    farthest = np.random.randint(0, N)  # 随机选择第一个点

    for i in range(num_samples):
        sampled[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)  # 更新最小距离
        farthest = np.argmax(distances)  # 找到最远的点

    return points[sampled]


def farthest_xyzrgb_point_sampling(
    points: List[Tuple[float, float, float, int, int, int]], num_samples: int
) -> List[Tuple[float, float, float, int, int, int]]:
    """
    使用 Farthest Point Sampling (FPS) 方法从点云中采样。

    Args:
        points (List[Tuple[float, float, float, int, int, int]]): 原始点云数据 (N, 6)，每个点包含 (x, y, z, r, g, b)
        num_samples (int): 采样点数

    Returns:
        List[Tuple[float, float, float, int, int, int]]: 采样后的点云数据 (num_samples, 6)
    """
    points = np.array(points)  # 转换为numpy数组
    N = points.shape[0]
    if num_samples >= N:
        return points.tolist()  # 转回列表

    sampled = np.zeros((num_samples,), dtype=np.int32)  # 存储采样点索引
    distances = np.ones((N,)) * 1e10  # 初始化每个点到已采样点的最小距离
    farthest = np.random.randint(0, N)  # 随机选择第一个点

    for i in range(num_samples):
        sampled[i] = farthest
        centroid = points[farthest, :]  # 现在可以正确索引
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)  # 更新最小距离
        farthest = np.argmax(distances)  # 找到最远的点

    return points[sampled].tolist()  # 转回列表


def filter_valid_depth(
    depth_image: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    过滤深度图中的有效点（非零且在合理范围内）

    Args:
        depth_image (np.ndarray): 深度图像
        min_depth (float): 最小有效深度（米）
        max_depth (float): 最大有效深度（米）
        depth_scale (float): 深度缩放因子，将深度值转换为米

    Returns:
        np.ndarray: 布尔掩码，标记有效深度点
    """
    # Convert depth to meters
    depth_meters = depth_image / depth_scale

    # Create mask for valid depth points
    valid_mask = (
        (depth_image > 0) & (depth_meters >= min_depth) & (depth_meters <= max_depth)
    )

    return valid_mask


def filter_points_dbscan(points_3d, colors, eps=0.02, min_samples=5):
    """使用DBSCAN聚类过滤点云，只保留最大聚类"""

    if len(points_3d) < min_samples:
        return points_3d, colors

    # 对点云进行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
    labels = clustering.labels_

    # 找到最大聚类
    if len(set(labels)) <= 1:  # 只有噪声或只有一个聚类
        if -1 in labels:  # 只有噪声
            return np.array([]), np.array([])
        return points_3d, colors

    # 统计各聚类的点数
    unique_labels = set(labels)
    max_cluster_label = -1
    max_cluster_size = 0

    for label in unique_labels:
        if label == -1:  # 跳过噪声
            continue
        cluster_size = np.sum(labels == label)
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            max_cluster_label = label

    # 提取最大聚类的点
    mask = labels == max_cluster_label
    filtered_points = points_3d[mask]
    filtered_colors = colors[mask]

    return filtered_points, filtered_colors


if __name__ == "__main__":
    import open3d as o3d
    import cv2
    import time

    # === 从图片读取 RGB-D 数据 ===
    H, W = 480, 640
    bgr_image = cv2.imread("/tmp/latest_bgr_image.png")
    depth_image = np.load("/tmp/latest_depth_image.npy")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # 相机内参与位姿
    fx = fy = 525.0
    cx = 319.5
    cy = 239.5
    intrinsic = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pose = np.eye(4)

    # 过滤有效深度
    valid_depth_mask = filter_valid_depth(depth_image, 0.1, 10, 1.0)
    all_uv = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)
    valid_uv = all_uv[valid_depth_mask.flatten()]

    time0 = time.time()
    # 随机采样 10000 个像素点
    sampled_uv = random_sampling(valid_uv, 100000)
    # sampled_uv = farthest_point_sampling(all_uv, 10000)
    time1 = time.time()
    u = sampled_uv[:, 0]
    v = sampled_uv[:, 1]
    z = depth_image[v, u]
    colors = rgb_image[v, u] / 255.0  # 归一化 RGB 值

    # 转换为世界坐标
    points_world = convert_from_uvd(u, v, z, intrinsic, pose)
    # points_world = convert_from_uvd_simu(
    #     u, v, z, pose, image_width=W, image_height=H, hfov=90.0, depth_scale=1.0
    # )
    time2 = time.time()
    # === 创建 open3d 点云对象 ===
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # === 显示 ===
    time3 = time.time()
    print(f"Sampling time: {time1 - time0:.4f} seconds")
    print(f"Conversion time: {time2 - time1:.4f} seconds")
    print(f"Total time: {time3 - time0:.4f} seconds")
    o3d.visualization.draw_geometries([pcd], window_name="Semantic World PointCloud")
