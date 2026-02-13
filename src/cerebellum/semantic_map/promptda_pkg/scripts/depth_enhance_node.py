import rospy
import torch
import numpy as np
import cv2
import os
import tempfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth
from promptda_pkg.srv import DepthEnhance, DepthEnhanceResponse


class DepthEnhanceNode:
    def __init__(self):
        rospy.init_node("depth_enhance_node")

        # 获取参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "depth-anything/prompt-depth-anything-vits"
        self.depth_scale = rospy.get_param("~depth_scale", 1.0)

        # 创建CV桥接器
        self.bridge = CvBridge()

        # 加载模型
        rospy.loginfo("Loading PromptDA model...")
        self.model = PromptDA.from_pretrained(self.model_name).to(self.device).eval()
        rospy.loginfo(f"Model loaded successfully on {self.device}")

        # 创建服务
        self.service = rospy.Service(
            "enhance_depth", DepthEnhance, self.handle_enhance_depth
        )
        rospy.loginfo("DepthEnhancer service initialized and ready")

    def load_image_from_array(self, image_array):
        """从numpy数组加载RGB图像到PromptDA格式"""
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

        # 使用PromptDA的加载函数
        tensor = load_image(temp_path)

        # 删除临时文件
        os.unlink(temp_path)
        return tensor

    def load_depth_from_array(self, depth_array):
        """从numpy数组加载深度图到PromptDA格式"""
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
            # 使用NPZ格式，这样load_depth不会执行除以1000操作
            np.savez(temp_path, depth=depth_array)

        # 使用PromptDA的加载函数
        tensor = load_depth(temp_path)

        # 删除临时文件
        os.unlink(temp_path)
        return tensor

    def handle_enhance_depth(self, req):
        try:
            # 删除断点
            # breakpoint()

            # 转换ROS消息到NumPy数组
            rgb_img = self.bridge.imgmsg_to_cv2(req.rgb_image, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(
                req.depth_image, desired_encoding="passthrough"
            )

            # 保存原始尺寸
            original_height, original_width = rgb_img.shape[:2]
            rospy.loginfo(
                f"Original image dimensions: {original_width}x{original_height}"
            )

            # 调整尺寸为14的倍数
            new_height = ((original_height + 13) // 14) * 14
            new_width = ((original_width + 13) // 14) * 14

            # 记录是否进行了调整
            resized = new_height != original_height or new_width != original_width

            if resized:
                rospy.loginfo(
                    f"Resizing images from {original_width}x{original_height} to {new_width}x{new_height}"
                )
                rgb_img = cv2.resize(rgb_img, (new_width, new_height))
                depth_img = cv2.resize(depth_img, (new_width, new_height))

            # 应用深度缩放
            depth_img = depth_img.astype(np.float32) / self.depth_scale

            # 转换为PyTorch张量并加载到设备
            rgb_tensor = self.load_image_from_array(rgb_img).to(self.device)
            depth_tensor = self.load_depth_from_array(depth_img).to(self.device)

            # 使用PromptDA预测增强深度
            with torch.no_grad():
                enhanced_depth = self.model.predict(rgb_tensor, depth_tensor)

            # 转换为NumPy数组并移除多余的维度 (1,1,H,W) -> (H,W)
            enhanced_depth_np = (
                enhanced_depth.cpu().numpy().squeeze() * self.depth_scale
            )

            # 如果调整过尺寸，恢复原始尺寸
            if resized:
                rospy.loginfo(
                    f"Resizing result back to original size: {original_width}x{original_height}"
                )
                enhanced_depth_np = cv2.resize(
                    enhanced_depth_np, (original_width, original_height)
                )

            # 转换为ROS消息
            enhanced_depth_msg = self.bridge.cv2_to_imgmsg(
                enhanced_depth_np.astype(np.float32), "32FC1"
            )
            enhanced_depth_msg.header = req.rgb_image.header

            return DepthEnhanceResponse(enhanced_depth=enhanced_depth_msg)

        except Exception as e:
            import traceback

            rospy.logerr(f"Error in depth enhancement: {e}")
            rospy.logerr(traceback.format_exc())
            return DepthEnhanceResponse(enhanced_depth=Image())


if __name__ == "__main__":
    try:
        node = DepthEnhanceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
