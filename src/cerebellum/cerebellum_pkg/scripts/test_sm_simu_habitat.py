#!/usr/bin/env python3
import os
import json
import rospy
import cv2
import glob
import numpy as np
import struct

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf2_ros
import geometry_msgs.msg
from tf import transformations

from coarse_localize.srv import Query as CoarseLocalizeQuery
from fine_grained_search.srv import Query as FineGrainedSearchQuery
from coarse_localize.srv import Show


class SMSimuHabitatTester:
    def __init__(self):
        # ROS Params
        rospy.init_node("sm_simu_habitat_tester")
        ## General
        self.dataset_path = rospy.get_param("~dataset_path", "/path/to/dataset")
        self.rgb_dir = os.path.join(self.dataset_path, "rgb")
        self.depth_dir = os.path.join(self.dataset_path, "depth")
        self.traj_dir = os.path.join(self.dataset_path, "trajectories")
        self.intri_file = os.path.join(self.dataset_path, "camera_intrinsics.json")
        self.bridge = CvBridge()
        self.max_sample_per_frame = rospy.get_param("~max_sample_per_frame", 5000)

        ## ROS publishers
        self.topic_rgb = rospy.get_param("~topic_rgb", "/habitat/rgb")
        self.topic_depth = rospy.get_param("~topic_depth", "/habitat/depth")
        self.topic_camera_info = rospy.get_param(
            "~topic_camera_info", "/habitat/camera_info"
        )
        self.topic_scene = rospy.get_param("~topic_scene", "/habitat/scene_pointcloud")
        self.rgb_pub = rospy.Publisher(self.topic_rgb, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(self.topic_depth, Image, queue_size=10)
        self.caminfo_pub = rospy.Publisher(
            self.topic_camera_info, CameraInfo, queue_size=10
        )
        self.scene_pub = rospy.Publisher(self.topic_scene, PointCloud2, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Load camera intrinsics
        self.cam_info = self.load_camera_info()

        # Service client
        rospy.wait_for_service("/memory_manager/memory_query")
        self.coarse_localize_client = rospy.ServiceProxy(
            "/memory_manager/memory_query", CoarseLocalizeQuery
        )
        rospy.wait_for_service("/memory_manager/memory_show")
        self.coarse_localize_show_client = rospy.ServiceProxy(
            "/memory_manager/memory_show", Show
        )
        # Timer to show memory markers
        rospy.Timer(rospy.Duration(1), self.memory_makers_show)

    def memory_makers_show(self, event):
        self.coarse_localize_show_client("all")

    def load_camera_info(self):
        with open(self.intri_file, "r") as f:
            intri = json.load(f)

        cam_info = CameraInfo()
        cam_info.width = intri["width"]
        cam_info.height = intri["height"]
        cam_info.K = [intri["fx"], 0, intri["cx"], 0, intri["fy"], intri["cy"], 0, 0, 1]
        cam_info.P = [
            intri["fx"],
            0,
            intri["cx"],
            0,
            0,
            intri["fy"],
            intri["cy"],
            0,
            0,
            0,
            1,
            0,
        ]
        return cam_info

    def publish_tf_frames(self, pose_list):
        """Publish TF coordinate system transformations"""
        try:
            now = rospy.Time.now()

            # pose_list = [x, y, z, qx, qy, qz, qw]
            x, y, z, qx, qy, qz, qw = pose_list

            # ---- map -> camera_link ----
            map2cam = geometry_msgs.msg.TransformStamped()
            map2cam.header.stamp = now
            map2cam.header.frame_id = "map"
            map2cam.child_frame_id = "camera_link"
            map2cam.transform.translation.x = x
            map2cam.transform.translation.y = y
            map2cam.transform.translation.z = z
            map2cam.transform.rotation.x = qx
            map2cam.transform.rotation.y = qy
            map2cam.transform.rotation.z = qz
            map2cam.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(map2cam)

            # ---- camera_link -> camera_rgb_optical_frame ----
            cam2rgb = geometry_msgs.msg.TransformStamped()
            cam2rgb.header.stamp = now
            cam2rgb.header.frame_id = "camera_link"
            cam2rgb.child_frame_id = "camera_rgb_optical_frame"
            q_optical = transformations.quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2)
            cam2rgb.transform.translation.x = 0.0
            cam2rgb.transform.translation.y = 0.0
            cam2rgb.transform.translation.z = 0.0
            cam2rgb.transform.rotation.x = q_optical[0]
            cam2rgb.transform.rotation.y = q_optical[1]
            cam2rgb.transform.rotation.z = q_optical[2]
            cam2rgb.transform.rotation.w = q_optical[3]
            self.tf_broadcaster.sendTransform(cam2rgb)

            # ---- camera_link -> camera_depth_optical_frame ----
            cam2depth = geometry_msgs.msg.TransformStamped()
            cam2depth.header.stamp = now
            cam2depth.header.frame_id = "camera_link"
            cam2depth.child_frame_id = "camera_depth_optical_frame"
            cam2depth.transform.translation.x = 0.0
            cam2depth.transform.translation.y = 0.0
            cam2depth.transform.translation.z = 0.0
            cam2depth.transform.rotation.x = q_optical[0]
            cam2depth.transform.rotation.y = q_optical[1]
            cam2depth.transform.rotation.z = q_optical[2]
            cam2depth.transform.rotation.w = q_optical[3]
            self.tf_broadcaster.sendTransform(cam2depth)

        except Exception as e:
            rospy.logerr(f"Error publishing TF: {e}")

    def depth_to_pointcloud(self, depth_img, rgb_img, pose_list):
        """
        depth_img: HxW (mm 或 m)
        rgb_img: HxWx3 (BGR)
        pose_list: [x,y,z,qx,qy,qz,qw] (map←camera_link)
        returns: Nx6 array: x,y,z,r,g,b
        """
        depth_img = depth_img.astype(np.float32) / 1000.0  # Convert to meters
        fx, fy = self.cam_info.K[0], self.cam_info.K[4]
        cx, cy = self.cam_info.K[2], self.cam_info.K[5]
        h, w = depth_img.shape

        # 生成像素坐标
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        zs = depth_img.astype(np.float32)
        # if depth likely in millimeters, convert to meters heuristically (optional)
        # if zs.max() > 100: zs = zs / 1000.0

        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy

        # 相机系下的点云 (N,3)
        pts_cam = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

        # RGB 颜色 (convert BGR->RGB)
        colors = rgb_img.reshape(-1, 3)[:, ::-1]  # BGR->RGB order

        # 去掉无效深度
        mask = zs.flatten() > 0
        if not np.any(mask):
            return np.zeros((0, 6), dtype=np.float32)

        pts_cam = pts_cam[mask]
        colors = colors[mask]

        # 如果点数量超过每帧采样上限，则随机下采样到 self.max_sample_per_frame
        if pts_cam.shape[0] > self.max_sample_per_frame:
            idx = np.random.choice(
                pts_cam.shape[0], int(self.max_sample_per_frame), replace=False
            )
            pts_cam = pts_cam[idx]
            colors = colors[idx]

        # 转换到 map 系
        x, y, z, qx, qy, qz, qw = pose_list
        T = transformations.quaternion_matrix([qx, qy, qz, qw])
        T[0:3, 3] = [x, y, z]
        pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)])
        pts_map = (T @ pts_h.T).T[:, :3]

        # 合并为 Nx6 (x,y,z,r,g,b)
        pts_rgb = np.hstack([pts_map, colors.astype(np.uint8)])
        return pts_rgb

    def publish_dataset(self):
        rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.png")))
        traj_files = sorted(glob.glob(os.path.join(self.traj_dir, "*.json")))

        assert (
            len(rgb_files) == len(depth_files) == len(traj_files)
        ), "Dataset files are inconsistent."
        n = len(rgb_files)
        scene_points = []

        for i in range(n):
            # timestamp from filename (去掉扩展名)
            stamp_str = os.path.basename(rgb_files[i]).split(".")[0]
            stamp = rospy.Time.now()  # 或者根据stamp_str转时间戳

            # ---- Trajectory Pose ----
            with open(traj_files[i], "r") as f:
                pose_list = json.load(f)
            self.publish_tf_frames(pose_list)

            # ---- RGB ----
            rgb_img = cv2.imread(rgb_files[i], cv2.IMREAD_COLOR)
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            self.rgb_pub.publish(rgb_msg)

            # ---- Depth ----
            depth_img = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="passthrough")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.depth_pub.publish(depth_msg)

            # ---- Camera Info ----
            caminfo = self.cam_info
            caminfo.header.stamp = stamp
            caminfo.header.frame_id = "camera_rgb_optical_frame"
            self.caminfo_pub.publish(caminfo)

            # ---- 点云采样 ----
            # breakpoint()
            frame_points = self.depth_to_pointcloud(depth_img, rgb_img, pose_list)
            if frame_points.shape[0] > 0:
                scene_points.append(frame_points)

            rospy.loginfo(f"Published sample {i+1}/{n}")
            rospy.sleep(0.1)  # 控制发布速率

        # ---- 全局点云发布 ----
        if len(scene_points) == 0:
            rospy.logwarn(
                "No scene points collected, skipping scene pointcloud publish."
            )
            rospy.loginfo("All dataset published.")
            return
        # breakpoint()
        all_points = np.vstack(scene_points)  # N x 6 (x,y,z,r,g,b)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            # pack RGB into a single 32-bit float (same memory layout as uint32)
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        # pack RGB into float representation expected by many ROS tools (PCL style)
        def pack_rgb_as_float(r, g, b):
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            return struct.unpack("f", struct.pack("I", rgb_int))[0]

        pc_data = [
            (
                float(p[0]),  # x
                float(p[1]),  # y
                float(p[2]),  # z
                pack_rgb_as_float(p[3], p[4], p[5]),  # packed rgb as float
            )
            for p in all_points
        ]

        pc_msg = pc2.create_cloud(header, fields, pc_data)
        self.scene_pub.publish(pc_msg)

        rospy.loginfo("All dataset published. Scene pointcloud published.")

    def query_after_dataset(self):
        user_query = input("请输入要查询的字符串: ")
        while user_query != "q":
            try:
                resp = self.coarse_localize_client(user_query)
                rospy.loginfo(f"Service response: {resp}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
            user_query = input("请输入要查询的字符串: ")


if __name__ == "__main__":
    publisher = SMSimuHabitatTester()
    publisher.publish_dataset()
    publisher.query_after_dataset()
