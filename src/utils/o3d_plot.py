import open3d as o3d
import numpy as np
import cv2


# rgb1 ----------------------------------------------------------------------------------------------------------------------------
# 读入图像（用 OpenCV）
img = cv2.imread("rgb1.jpg")
# add black border
img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB
h, w, _ = img.shape

# --- 构建点云（每个像素变成一个点） ---
xs, ys = np.meshgrid(np.linspace(0, 0.4, w), np.linspace(0, 0.3, h))
zs = np.zeros_like(xs)  # 全部放在 z=0 平面
points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

colors = img.reshape(-1, 3) / 255.0  # 归一化到 [0,1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 平移到投影平面位置
pcd.translate([0.8, -0.15, 0.1])

# --- 机器人眼睛位置 ---
eye = np.array([1.0, 0.0, -0.2])  # 可以调整

# --- 在眼睛位置加坐标系 ---
eye_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
eye_coord.translate(eye)  # 移动到眼睛位置
R = eye_coord.get_rotation_matrix_from_axis_angle([0, -np.pi / 2, 0])
eye_coord.rotate(R, center=eye)
R = eye_coord.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 2])
eye_coord.rotate(R, center=eye)

# --- 投影平面四个角 ---
corners = np.array(
    [
        [0.8, -0.15, 0.1],  # 左下
        [1.2, -0.15, 0.1],  # 右下
        [0.8, 0.15, 0.1],  # 左上
        [1.2, 0.15, 0.1],  # 右上
    ]
)

# --- 投影线 (LineSet) ---
points_all = np.vstack([eye, corners])
lines = [[0, 1], [0, 2], [0, 3], [0, 4]]  # eye -> 每个角
colors = [[0, 0, 0] for _ in lines]  # 黑色线

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points_all),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
# # rgb2 ----------------------------------------------------------------------------------------------------------------------------
# img2 = cv2.imread("rgb2.jpg")
# img2 = cv2.copyMakeBorder(img2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# h2, w2, _ = img2.shape
# xs2, ys2 = np.meshgrid(np.linspace(0, 0.4, w2), np.linspace(0, 0.3, h2))
# zs2 = np.zeros_like(xs2)  # 全部放在 z=0 平面
# points2 = np.stack((xs2, ys2, zs2), axis=-1).reshape(-1, 3)
# colors2 = (img2.reshape(-1, 3) / 255.0)  # 归一化到 [0,1]
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(points2)
# pcd2.colors = o3d.utility.Vector3dVector(colors2)
# pcd2.translate([0.8, -0.15, 0.1])
# eye2 = np.array([1.0, 0.0, -0.2])  # 可以调整
# --- 显示 ---
o3d.visualization.draw_geometries([pcd, eye_coord, line_set])
