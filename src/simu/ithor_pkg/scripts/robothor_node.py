import numpy as np
from ai2thor.controller import Controller
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
import geometry_msgs.msg
from tf import transformations
import struct

# from asm.scripts.asm_lib.pcd_utils import convert_from_uvd
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


class RoboTHORNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("robothor_simulator", anonymous=True)

        # Create CV Bridge instance
        self.bridge = CvBridge()

        # Initialize image publishers
        topic_rgb = rospy.get_param("~topic_rgb", "/robothor/rgb")
        topic_depth = rospy.get_param("~topic_depth", "/robothor/depth")
        topic_camera_info = rospy.get_param(
            "~topic_camera_info", "/robothor/camera_info"
        )
        self.enable_sample_pcd = rospy.get_param("~enable_sample_pcd", False)

        self.rgb_pub = rospy.Publisher(topic_rgb, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(topic_depth, Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher(
            topic_camera_info, CameraInfo, queue_size=10
        )
        # 添加点云发布器
        self.sample_pcd_pub = rospy.Publisher(
            "/robothor/sample_pcd", PointCloud2, queue_size=1
        )

        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize control mapping for keyboard commands
        self.controls = {
            "W": "MoveAhead",
            "A": "RotateLeft",
            "S": "MoveBack",
            "D": "RotateRight",
        }

        # Subscribe to keyboard command topic
        rospy.Subscriber("/ithor/keyboard_cmd", String, self.keyboard_callback)

        # Initialize AI2-THOR controller using locobot mode
        self.FRAME_WIDTH = rospy.get_param("~frame_width", 640)
        self.FRAME_HEIGHT = rospy.get_param("~frame_height", 480)
        self.FRAME_FOV = rospy.get_param("~frame_fov", 60.0)
        self.FRAME_GRID_SIZE = rospy.get_param("~frame_grid_size", 0.1)
        self.FRAME_VISIBILITY_DISTANCE = rospy.get_param(
            "~frame_visibility_distance", 1.5
        )
        self.SCENE_NAME = rospy.get_param("~scene_name", "FloorPlan1")
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=self.FRAME_VISIBILITY_DISTANCE,
            scene=self.SCENE_NAME,
            gridSize=self.FRAME_GRID_SIZE,
            movementGaussianSigma=0.0,
            rotateStepDegrees=90,
            rotateGaussianSigma=0.0,
            renderDepthImage=True,
            renderInstanceSegmentation=False,
            width=self.FRAME_WIDTH,
            height=self.FRAME_HEIGHT,
            fieldOfView=self.FRAME_FOV,
        )

        self.origin_simu_pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
        }  # m & rad
        self.simu_pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
        }  # m & rad
        self.cam_info = None
        self.ros_pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
        }  # m & rad
        # Initialize and get first event
        self.event = self.controller.step(action="Pass")
        self.init_robot(self.event)

        # Running state
        self.running = True

        rospy.loginfo("RoboTHOR simulator node started")

    def init_robot(self, first_event):
        """Initialize robot with the first event"""
        agent_metadata = first_event.metadata["agent"]
        self.origin_simu_pose["x"] = agent_metadata["position"]["x"]
        self.origin_simu_pose["y"] = agent_metadata["position"]["y"]
        self.origin_simu_pose["z"] = agent_metadata["position"]["z"]
        self.origin_simu_pose["roll"] = np.deg2rad(agent_metadata["rotation"]["x"])
        self.origin_simu_pose["pitch"] = np.deg2rad(agent_metadata["rotation"]["y"])
        self.origin_simu_pose["yaw"] = np.deg2rad(agent_metadata["rotation"]["z"])
        self.cam_info = self.get_default_camera_info(
            width=self.FRAME_WIDTH,
            height=self.FRAME_HEIGHT,
            fov=self.FRAME_FOV,
        )
        rospy.loginfo("Robot initialized with first event")

    def get_default_camera_info(self, width=640, height=480, fov=90.0):
        """根据视场角计算相机内参"""
        cam_info = CameraInfo()
        cam_info.width = width
        cam_info.height = height

        # 从FOV计算焦距: f = (width/2) / tan(FOV/2)
        fx = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        # 设置相机内参 K, P, R, D
        cam_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        cam_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        cam_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        cam_info.D = [0, 0, 0, 0, 0]
        cam_info.distortion_model = "plumb_bob"
        cam_info.header.frame_id = "camera_rgb_optical_frame"
        return cam_info

    def renew_simu_pose(self):
        cur_agent_metadata = self.event.metadata["agent"]
        cur_position = cur_agent_metadata["position"]
        cur_rotation = cur_agent_metadata["rotation"]
        cur_camera_horizon = cur_agent_metadata["cameraHorizon"]
        """Calculate relative pose from origin"""
        self.simu_pose["x"] = cur_position["x"] - self.origin_simu_pose["x"]
        self.simu_pose["y"] = cur_position["y"] - self.origin_simu_pose["y"]
        self.simu_pose["z"] = cur_position["z"] - self.origin_simu_pose["z"]
        self.simu_pose["roll"] = (
            np.deg2rad(cur_rotation["x"]) - self.origin_simu_pose["roll"]
        )
        self.simu_pose["pitch"] = (
            np.deg2rad(cur_rotation["y"]) - self.origin_simu_pose["pitch"]
        )
        self.simu_pose["yaw"] = (
            np.deg2rad(cur_rotation["z"]) - self.origin_simu_pose["yaw"]
        )

    def renew_ros_pose(self):
        """Convert AI2-THOR coordinates to ROS coordinates"""
        self.ros_pose["x"] = -self.simu_pose["x"]
        self.ros_pose["y"] = self.simu_pose["y"]
        self.ros_pose["z"] = self.simu_pose["z"]
        self.ros_pose["roll"] = self.simu_pose["roll"]
        self.ros_pose["pitch"] = self.simu_pose["yaw"]
        self.ros_pose["yaw"] = -self.simu_pose["pitch"]

    def publish_tf_frames(self):
        """Publish TF coordinate system transformations"""
        try:
            now = rospy.Time.now()

            # 创建变换消息：map -> camera_link
            map2cam = geometry_msgs.msg.TransformStamped()
            map2cam.header.stamp = now
            map2cam.header.frame_id = "map"
            map2cam.child_frame_id = "camera_link"
            map2cam.transform.translation.x = self.ros_pose["x"]
            map2cam.transform.translation.y = self.ros_pose["y"]
            map2cam.transform.translation.z = self.ros_pose["z"]
            quat = transformations.quaternion_from_euler(
                self.ros_pose["roll"],
                self.ros_pose["pitch"],
                self.ros_pose["yaw"],
            )
            map2cam.transform.rotation.x = quat[0]
            map2cam.transform.rotation.y = quat[1]
            map2cam.transform.rotation.z = quat[2]
            map2cam.transform.rotation.w = quat[3]
            # 发布 map -> camera_link 变换
            self.tf_broadcaster.sendTransform(map2cam)

            # 创建变换消息：camera_link -> camera_rgb_optical_frame
            cam2cam_rgb = geometry_msgs.msg.TransformStamped()
            cam2cam_rgb.header.stamp = now
            cam2cam_rgb.header.frame_id = "camera_link"
            cam2cam_rgb.child_frame_id = "camera_rgb_optical_frame"
            q_optical = transformations.quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2)
            # q_optical = transformations.quaternion_from_euler(0, 0, 0)
            cam2cam_rgb.transform.translation.x = 0.0
            cam2cam_rgb.transform.translation.y = 0.0
            cam2cam_rgb.transform.translation.z = 0.0
            cam2cam_rgb.transform.rotation.x = q_optical[0]
            cam2cam_rgb.transform.rotation.y = q_optical[1]
            cam2cam_rgb.transform.rotation.z = q_optical[2]
            cam2cam_rgb.transform.rotation.w = q_optical[3]
            # 发布 camera -> camera_rgb_optical_frame 变换
            self.tf_broadcaster.sendTransform(cam2cam_rgb)

            # 创建变换消息：camera_link -> camera_depth_optical_frame
            cam2cam_depth = geometry_msgs.msg.TransformStamped()
            cam2cam_depth.header.stamp = now
            cam2cam_depth.header.frame_id = "camera_link"
            cam2cam_depth.child_frame_id = "camera_depth_optical_frame"
            q_depth = transformations.quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2)
            # q_depth = transformations.quaternion_from_euler(0, 0, 0)
            cam2cam_depth.transform.translation.x = 0.0
            cam2cam_depth.transform.translation.y = 0.0
            cam2cam_depth.transform.translation.z = 0.0
            cam2cam_depth.transform.rotation.x = q_depth[0]
            cam2cam_depth.transform.rotation.y = q_depth[1]
            cam2cam_depth.transform.rotation.z = q_depth[2]
            cam2cam_depth.transform.rotation.w = q_depth[3]
            # 发布 camera -> camera_depth_optical_frame 变换
            self.tf_broadcaster.sendTransform(cam2cam_depth)

        except KeyError as e:
            rospy.logerr(f"Could not find required field in metadata: {e}")
        except Exception as e:
            rospy.logerr(f"Error publishing TF: {e}")

    def common_pub(self):
        """Process event, get image data and publish"""
        try:
            rgb_image = self.event.frame
            depth_image = self.event.depth_frame
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")

            # Publish rgb + depth messages
            now = rospy.Time.now()
            rgb_msg.header.stamp = now
            depth_msg.header.stamp = now
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)

            # Update camera info
            self.cam_info.header.stamp = now
            self.camera_info_pub.publish(self.cam_info)

            # 先更新机器人位姿
            self.renew_simu_pose()
            self.renew_ros_pose()

            # Publish TF transformations
            self.publish_tf_frames()

            # 如果启用了点云生成，则生成点云并发布
            if self.enable_sample_pcd:
                point_cloud = self.generate_point_cloud(
                    rgb_image, depth_image, num_points=100000
                )
                if point_cloud:
                    self.sample_pcd_pub.publish(point_cloud)

        except Exception as e:
            rospy.logerr(f"Image publishing failed: {e}")

    def keyboard_callback(self, msg):
        """Handle keyboard commands"""
        try:
            key = msg.data
            if key in self.controls:
                action = self.controls[key]
                if action in ["MoveAhead", "MoveBack"]:
                    self.event = self.controller.step(action=action, moveMagnitude=0.1)
                elif action in ["RotateLeft", "RotateRight"]:
                    self.event = self.controller.step(action=action, degrees=15)
                else:
                    self.event = self.controller.step(action=action)
                self.renew_simu_pose()
                self.renew_ros_pose()
                self.controller.step(action="Done")
                rospy.logwarn(
                    f"self.simu_pose: {self.simu_pose}\nself.ros_pose: {self.ros_pose}"
                )
            else:
                rospy.loginfo(f"Unknown key command: {key}")
        except Exception as e:
            rospy.logerr(f"Keyboard callback error: {str(e)}")

    def run(self):
        """Run main loop"""
        rate = rospy.Rate(30)  # 30Hz update frequency

        while not rospy.is_shutdown() and self.running:
            # Periodically publish images regardless of keyboard input
            self.common_pub()
            rate.sleep()

        # Clean up resources
        self.controller.stop()
        rospy.loginfo("RoboTHOR simulator node shut down")

    def generate_point_cloud(self, rgb_image, depth_image, num_points=10000):
        """从RGB和深度图像生成点云，投影到camera_rgb_optical_frame坐标系

        Args:
            rgb_image: RGB格式的彩色图像
            depth_image: 深度图像(单位:米)
            num_points: 采样的点云数量

        Returns:
            点云数据（PointCloud2格式）
        """
        try:
            # 1. 在有效深度区域中随机采样点
            valid_pixels = np.where(depth_image > 0)
            if len(valid_pixels[0]) == 0:
                rospy.logwarn("No valid depth pixels found for point cloud")
                return None

            # 采样点
            if len(valid_pixels[0]) <= num_points:
                # 使用所有有效像素
                indices = np.arange(len(valid_pixels[0]))
            else:
                # 随机采样点
                indices = np.random.choice(
                    len(valid_pixels[0]), num_points, replace=False
                )

            # 获取采样点的行列索引和深度值
            v = valid_pixels[0][indices]  # y坐标
            u = valid_pixels[1][indices]  # x坐标
            d = depth_image[v, u]  # 深度值

            # 2. 准备相机内参矩阵 (4x4)
            fx = self.cam_info.K[0]
            fy = self.cam_info.K[4]
            cx = self.cam_info.K[2]
            cy = self.cam_info.K[5]

            intr = np.array(
                [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

            # 3. 创建单位矩阵作为位姿矩阵
            # 因为我们生成的点云就位于camera_rgb_optical_frame中
            pose = np.eye(4)

            # 4. 使用convert_from_uvd函数进行坐标转换
            # 深度值已经是米单位，所以depth_scale=1.0
            points3d = convert_from_uvd(u, v, d, intr, pose, depth_scale=1.0)

            # 5. 获取对应的RGB颜色并创建点云数据
            cloud_points = []
            for i in range(len(points3d)):
                x, y, z = points3d[i]
                r, g, b = rgb_image[v[i], u[i]]
                # 打包RGB值为一个32位整数
                rgb_packed = struct.unpack("I", struct.pack("BBBB", b, g, r, 255))[0]
                cloud_points.append([x, y, z, rgb_packed])

            # 6. 创建PointCloud2消息
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.UINT32, 1),
            ]

            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_rgb_optical_frame"  # 点云坐标系与图像一致

            pc = pc2.create_cloud(header, fields, cloud_points)
            return pc

        except Exception as e:
            rospy.logerr(f"Error generating point cloud: {e}")
            import traceback

            traceback.print_exc()
            return None


if __name__ == "__main__":
    try:
        robothor_node = RoboTHORNode()
        robothor_node.run()
    except Exception as e:
        rospy.logerr(f"RoboTHOR node encountered an error: {e}")
