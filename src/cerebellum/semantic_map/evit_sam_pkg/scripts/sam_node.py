import rospy
import numpy as np
import cv2
import torch
from cv_bridge import CvBridge
from evit_sam_pkg.srv import EvitSamSegmentation, EvitSamSegmentationResponse

# 引入两个引擎
from evit_sam_engine import EvitSAMEngine
from fast_sam_engine import FastSAMEngine


class SAMNode:
    def __init__(self):
        rospy.init_node("sam_node")
        device = rospy.get_param("~device", "cuda")
        evit_ckpt = rospy.get_param(
            "~evit_sam_checkpoint", "weights/efficientvit_sam_l1.pt"
        )
        evit_model = rospy.get_param("~evit_sam_model", "efficientvit-sam-l1")
        fast_ckpt = rospy.get_param("~fast_sam_checkpoint", "weights/FastSAM-x.pt")

        # rospy.loginfo("Loading both EfficientViT-SAM and FastSAM...")

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            rospy.logerr("No GPU available, falling back to CPU.")
            self.device = torch.device("cpu")

        self.engines = {
            "evit-sam": EvitSAMEngine(
                model_type=evit_model, ckpt_path=evit_ckpt, device=self.device
            ),
            "fast-sam": FastSAMEngine(
                model_path=fast_ckpt, device=self.device, imgsz=1024, conf=0.4, iou=0.9
            ),
        }

        self.device = device
        self.cv_bridge = CvBridge()

        # 初始化服务
        self.service = rospy.Service(
            "sam_segmentation", EvitSamSegmentation, self.callback
        )
        # rospy.loginfo("SAM Segmentation Service Ready.")

    def callback(self, request):
        model_name = request.model.lower()
        mode = request.mode.lower()

        if model_name not in self.engines:
            rospy.logerr(f"Unsupported model type: {model_name}")
            return EvitSamSegmentationResponse()

        engine = self.engines[model_name]

        # 解码图像
        img = self.cv_bridge.imgmsg_to_cv2(request.color_image, desired_encoding="bgr8")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # start_time = rospy.Time.now()

        try:
            if mode == "full":
                masks = engine.segment_everything(rgb_img)
                if masks is None:
                    return EvitSamSegmentationResponse()
            elif mode == "bbox":
                # reshape bounding boxes
                boxes_data = np.array(request.boxes)
                xyxy = boxes_data.reshape(-1, 4)

                if xyxy.size == 0:
                    rospy.logwarn("Empty box input received.")
                    masks = np.empty(
                        (0, rgb_img.shape[0], rgb_img.shape[1]), dtype=bool
                    )
                else:
                    masks = engine.segment_by_boxes(rgb_img, xyxy)

            else:
                raise ValueError(f"Unsupported mode: {mode}")
            # 可视化叠加
            masked_image = rgb_img.copy()
            for mask in masks:
                masked_image[mask.astype(bool)] = [
                    np.random.randint(0, 256) for _ in range(3)
                ]  # random color overlay

            # seg_time = (rospy.Time.now() - start_time).to_sec()
            # rospy.loginfo(
            #     f"[{model_name}] Segmentation done in {seg_time:.3f} s, {len(masks)} masks"
            # )

            # 构造响应
            response = EvitSamSegmentationResponse()
            response.seg_masks = []

            for i, mask in enumerate(masks):
                mask_8bit = (mask * 255).astype(np.uint8)
                mask_msg = self.cv_bridge.cv2_to_imgmsg(mask_8bit, encoding="mono8")
                mask_msg.header.stamp = rospy.Time.now()
                mask_msg.header.frame_id = f"mask_{i}"
                response.seg_masks.append(mask_msg)

            masked_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            response.masked_frame = self.cv_bridge.cv2_to_imgmsg(
                masked_bgr, encoding="bgr8"
            )

        except Exception as e:
            rospy.logerr(f"[EvitSamServer] Error: {str(e)}")
            response = EvitSamSegmentationResponse()
            response.seg_masks = []
            response.masked_frame = request.color_image

        torch.cuda.empty_cache()
        return response


if __name__ == "__main__":
    try:
        node = SAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error starting SAMNode: {e}")
