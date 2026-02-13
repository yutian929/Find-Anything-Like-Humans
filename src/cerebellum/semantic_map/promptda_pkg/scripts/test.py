import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
import cv2
import numpy as np
from promptda_pkg.srv import DepthEnhance


class DepthEnhanceTest:
    def __init__(self):
        rospy.init_node("depth_enhance_test")

        # 获取参数
        self.rgb_topic = rospy.get_param("~rgb_topic", "/ai2thor/rgb")
        self.depth_topic = rospy.get_param("~depth_topic", "/ai2thor/depth")

        # 创建 CV 桥接器
        self.bridge = CvBridge()

        # 等待深度增强服务
        rospy.loginfo("Waiting for enhance_depth service...")
        rospy.wait_for_service("enhance_depth")
        self.enhance_service = rospy.ServiceProxy("enhance_depth", DepthEnhance)
        rospy.loginfo("Connected to enhance_depth service")

        # 创建可视化发布器
        self.rgb_pub = rospy.Publisher("~rgb_view", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("~depth_view", Image, queue_size=1)
        self.enhanced_pub = rospy.Publisher("~enhanced_view", Image, queue_size=1)
        self.combined_pub = rospy.Publisher("~combined_view", Image, queue_size=1)

        # 设置同步订阅器
        self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)

        # 使用近似时间同步策略
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1
        )
        ts.registerCallback(self.callback)

        rospy.loginfo(
            f"Test node initialized, listening to {self.rgb_topic} and {self.depth_topic}"
        )

    def callback(self, rgb_msg, depth_msg):
        try:
            # 转换ROS消息到OpenCV格式
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg)

            # 记录处理开始时间
            start_time = rospy.Time.now()

            # 调用深度增强服务
            response = self.enhance_service(rgb_msg, depth_msg)
            enhanced_img = self.bridge.imgmsg_to_cv2(response.enhanced_depth)

            # 计算处理时间
            process_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"Depth enhancement completed in {process_time:.3f} seconds")

            # 创建可视化图像
            # 为了更好的可视化，对深度图进行归一化处理
            depth_norm = self.normalize_depth(depth_img)
            enhanced_norm = self.normalize_depth(enhanced_img)

            # 发布单独的可视化结果
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, "rgb8"))
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_norm, "mono8"))
            self.enhanced_pub.publish(self.bridge.cv2_to_imgmsg(enhanced_norm, "mono8"))

            # 创建组合可视化
            h, w = rgb_img.shape[:2]
            combined = np.zeros((h, w * 3, 3), dtype=np.uint8)

            # 转换灰度图为彩色，用于显示
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            enhanced_color = cv2.applyColorMap(enhanced_norm, cv2.COLORMAP_JET)

            # 将RGB图像从RGB转换为BGR (OpenCV格式)
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # 组合图像
            combined[:, :w] = rgb_bgr
            combined[:, w : 2 * w] = depth_color
            combined[:, 2 * w :] = enhanced_color

            # 添加标签
            cv2.putText(
                combined,
                "RGB",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined,
                "Original Depth",
                (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined,
                "Enhanced Depth",
                (2 * w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # 发布组合可视化
            combined_msg = self.bridge.cv2_to_imgmsg(combined, "bgr8")
            combined_msg.header = rgb_msg.header
            self.combined_pub.publish(combined_msg)

        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

    def normalize_depth(self, depth_img):
        """将深度图归一化到0-255范围以便可视化"""
        min_val = np.min(depth_img)
        max_val = np.max(depth_img)
        if max_val > min_val:
            depth_norm = ((depth_img - min_val) / (max_val - min_val) * 255).astype(
                np.uint8
            )
        else:
            depth_norm = np.zeros_like(depth_img, dtype=np.uint8)
        return depth_norm


if __name__ == "__main__":
    try:
        node = DepthEnhanceTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
