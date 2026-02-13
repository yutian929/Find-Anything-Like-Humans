import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from evit_sam_pkg.srv import EvitSamSegmentation, EvitSamSegmentationRequest
from yolo_world_pkg.srv import YoloDetection, YoloDetectionRequest
from cv_bridge import CvBridge
import numpy as np
import cv2
import os


class DetSegTester:
    def __init__(self):
        rospy.init_node("test_detseg_node")
        self.cv_bridge = CvBridge()

        # 参数设置
        self.prompt = rospy.get_param(
            "~prompt",
            ["sofa", "chair", "table", "cup", "cola", "human", "refrigerator", "vase"],
        )  # 添加更多类别
        self.model = rospy.get_param("~seg_model", "evit")  # "evit" or "fast"
        self.image_path = rospy.get_param("~image_path", "/home/yutian/YanBot/328.png")
        self.debug_mode = rospy.get_param("~debug_mode", True)  # 是否开启debug模式

        # 等待服务
        rospy.loginfo("Waiting for YOLO detection service...")
        rospy.wait_for_service("yolo_detection")
        self.det_client = rospy.ServiceProxy("yolo_detection", YoloDetection)
        rospy.loginfo("YOLO service connected.")

        rospy.loginfo("Waiting for SAM segmentation service...")
        rospy.wait_for_service("sam_segmentation")
        self.seg_client = rospy.ServiceProxy("sam_segmentation", EvitSamSegmentation)
        rospy.loginfo("SAM service connected.")

        # 等待1秒后开始处理
        rospy.loginfo("Services ready. Waiting 1 second before processing...")
        rospy.sleep(1.0)

        # 处理图像
        self.process_image()

    def process_image(self):
        """读取并处理指定的图像文件"""
        if not os.path.exists(self.image_path):
            rospy.logerr(f"Image file not found: {self.image_path}")
            return

        rospy.loginfo(f"Reading image from: {self.image_path}")

        # 读取图像
        cv_image = cv2.imread(self.image_path)
        if cv_image is None:
            rospy.logerr(f"Failed to load image: {self.image_path}")
            return
        cv_image = cv2.resize(cv_image, (640, 480))  # 调整图像大小以适应处理

        # 转换为ROS Image消息
        ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")

        rospy.loginfo("Processing image...")

        if self.debug_mode:
            # 1. 显示原始RGB图像
            cv2.imshow("1. Original RGB", cv_image)
            cv2.waitKey(1000)  # 显示1秒

        # Step 1: Call detection
        det_req = YoloDetectionRequest()
        det_req.color_image = ros_image
        det_req.prompt = self.prompt
        det_res = self.det_client(det_req)

        boxes_np = self.parse_boxes(det_res.boxes)
        if boxes_np.shape[0] == 0:
            rospy.logwarn("No detections found. Skipping segmentation.")
            if self.debug_mode:
                cv2.destroyAllWindows()
            return

        rospy.loginfo(f"Detected {len(det_res.labels)} objects: {det_res.labels}")

        if self.debug_mode:
            # 2. 显示YOLO检测结果总图
            try:
                # Try different encoding formats for the annotated frame
                try:
                    annotated_cv = self.cv_bridge.imgmsg_to_cv2(
                        det_res.annotated_frame, "bgr8"
                    )
                except Exception:
                    try:
                        annotated_cv = self.cv_bridge.imgmsg_to_cv2(
                            det_res.annotated_frame, "rgb8"
                        )
                        annotated_cv = cv2.cvtColor(annotated_cv, cv2.COLOR_RGB2BGR)
                    except Exception:
                        annotated_cv = self.cv_bridge.imgmsg_to_cv2(
                            det_res.annotated_frame
                        )

                cv2.imshow("2. YOLO Detection Results", annotated_cv)
                cv2.waitKey(1000)

                # 3. 显示每个检测框对应的图像
                self.show_individual_boxes(cv_image, boxes_np, det_res.labels)

            except Exception as e:
                rospy.logwarn(f"Failed to display detection results: {e}")

        # Step 2: Call segmentation
        seg_req = EvitSamSegmentationRequest()
        seg_req.model = self.model
        seg_req.mode = "bbox"
        seg_req.color_image = ros_image
        seg_req.boxes = self.pack_boxes(boxes_np)

        seg_res = self.seg_client(seg_req)
        rospy.loginfo(f"Received {len(seg_res.seg_masks)} segmentation masks.")

        if self.debug_mode:
            # 4. 显示分割结果总图
            try:
                # Try different encoding formats for the masked frame
                try:
                    masked_cv = self.cv_bridge.imgmsg_to_cv2(
                        seg_res.masked_frame, "bgr8"
                    )
                except Exception:
                    try:
                        masked_cv = self.cv_bridge.imgmsg_to_cv2(
                            seg_res.masked_frame, "rgb8"
                        )
                        masked_cv = cv2.cvtColor(masked_cv, cv2.COLOR_RGB2BGR)
                    except Exception:
                        masked_cv = self.cv_bridge.imgmsg_to_cv2(seg_res.masked_frame)

                cv2.imshow("4. SAM Segmentation Results", masked_cv)
                cv2.waitKey(1000)

                # 5. 显示每个掩码对应的图像
                self.show_individual_masks(cv_image, seg_res.seg_masks, det_res.labels)

            except Exception as e:
                rospy.logwarn(f"Failed to display segmentation results: {e}")

        if self.debug_mode:
            rospy.loginfo("Press 'q' to continue or ESC to exit...")
            key = cv2.waitKey(0)
            if key == 27:  # ESC键
                rospy.signal_shutdown("User requested shutdown")
            cv2.destroyAllWindows()

        rospy.loginfo("Image processing completed.")

    def show_individual_boxes(self, image, boxes, labels):
        """显示每个检测框对应的图像区域"""
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)

            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # 裁剪图像区域
            cropped = image[y1:y2, x1:x2].copy()

            if cropped.size > 0:
                # 在裁剪图像上绘制边框
                cv2.rectangle(
                    cropped,
                    (0, 0),
                    (cropped.shape[1] - 1, cropped.shape[0] - 1),
                    (0, 255, 0),
                    2,
                )

                # 添加标签文本
                cv2.putText(
                    cropped,
                    f"{label}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # 调整显示大小
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    # 如果图像太小，放大显示
                    if max(cropped.shape[:2]) < 100:
                        scale = 100 / max(cropped.shape[:2])
                        new_size = (
                            int(cropped.shape[1] * scale),
                            int(cropped.shape[0] * scale),
                        )
                        cropped = cv2.resize(
                            cropped, new_size, interpolation=cv2.INTER_NEAREST
                        )

                    cv2.imshow(f"3.{i+1}. Box: {label}", cropped)
                    cv2.waitKey(800)  # 每个框显示0.8秒

    def show_individual_masks(self, image, masks, labels):
        """显示每个掩码对应的图像"""
        for i, (mask_msg, label) in enumerate(zip(masks, labels)):
            try:
                # 解析掩码数据 - masks are Image messages, not Float32MultiArray
                if hasattr(mask_msg, "data"):
                    # Convert ROS Image message to OpenCV format
                    try:
                        # Try to convert as grayscale image
                        mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "mono8")
                    except Exception:
                        try:
                            # Try other formats if mono8 fails
                            mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "8UC1")
                        except Exception:
                            try:
                                mask = self.cv_bridge.imgmsg_to_cv2(mask_msg)
                                # Convert to grayscale if it's a color image
                                if len(mask.shape) == 3:
                                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            except Exception as e:
                                rospy.logwarn(
                                    f"Failed to convert mask {i} to OpenCV format: {e}"
                                )
                                continue
                else:
                    rospy.logwarn(f"Mask {i} has no data attribute")
                    continue

                if mask is None or mask.size == 0:
                    rospy.logwarn(f"Mask {i} has empty data")
                    continue

                # Ensure mask is binary (0 or 255)
                mask = (mask > 127).astype(np.uint8) * 255

                # 将掩码应用到原图像
                masked_image = image.copy()

                # 创建彩色掩码
                color_mask = np.zeros_like(image)
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
                color_mask[mask > 0] = color

                # 混合原图像和掩码
                alpha = 0.6
                masked_image = cv2.addWeighted(
                    masked_image, 1 - alpha, color_mask, alpha, 0
                )

                # 只显示掩码区域
                mask_only = image.copy()
                mask_only[mask == 0] = [0, 0, 0]  # 将非掩码区域设为黑色

                # 获取掩码的边界框用于裁剪
                coords = np.column_stack(np.where(mask > 0))
                if len(coords) > 0:
                    y1, x1 = coords.min(axis=0)
                    y2, x2 = coords.max(axis=0)

                    # 添加一些边距
                    margin = 10
                    y1 = max(0, y1 - margin)
                    x1 = max(0, x1 - margin)
                    y2 = min(mask.shape[0], y2 + margin)
                    x2 = min(mask.shape[1], x2 + margin)

                    # 裁剪掩码区域
                    cropped_masked = masked_image[y1:y2, x1:x2]
                    cropped_mask_only = mask_only[y1:y2, x1:x2]

                    if cropped_masked.size > 0:
                        # 添加标签
                        cv2.putText(
                            cropped_masked,
                            f"{label}",
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            cropped_mask_only,
                            f"{label}",
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                        # 调整显示大小
                        if max(cropped_masked.shape[:2]) < 150:
                            scale = 150 / max(cropped_masked.shape[:2])
                            new_size = (
                                int(cropped_masked.shape[1] * scale),
                                int(cropped_masked.shape[0] * scale),
                            )
                            cropped_masked = cv2.resize(
                                cropped_masked,
                                new_size,
                                interpolation=cv2.INTER_NEAREST,
                            )
                            cropped_mask_only = cv2.resize(
                                cropped_mask_only,
                                new_size,
                                interpolation=cv2.INTER_NEAREST,
                            )

                        # 并排显示混合图像和纯掩码
                        combined = np.hstack([cropped_masked, cropped_mask_only])
                        cv2.imshow(f"5.{i+1}. Mask: {label} (Overlay | Pure)", combined)
                        cv2.waitKey(800)  # 每个掩码显示0.8秒
                else:
                    rospy.logwarn(f"Mask {i} for {label} has no positive pixels")

            except Exception as e:
                rospy.logwarn(f"Failed to display mask {i}: {e}")

    def parse_boxes(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) == 0:
            return np.empty((0, 4), dtype=np.float32)

        box_num = msg.layout.dim[0].size
        return data.reshape((box_num, 4))

    def pack_boxes(self, boxes):
        """
        将 Nx4 numpy box 转换为 Float32MultiArray
        """
        msg = Float32MultiArray()
        msg.data = boxes.astype(np.float32).flatten().tolist()

        dim = MultiArrayDimension()
        dim.label = "boxes"
        dim.size = boxes.shape[0]
        dim.stride = boxes.size
        msg.layout.dim = [dim]
        return msg


if __name__ == "__main__":
    try:
        tester = DetSegTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
