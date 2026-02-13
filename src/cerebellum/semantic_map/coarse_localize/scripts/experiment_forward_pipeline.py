#!/usr/bin/env python3
import threading
import time
import numpy as np
from typing import List, Tuple

import message_filters
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from caod_pkg.srv import CAOD, CAODRequest, CAODResponse
from clip_pkg.srv import CLIP, CLIPRequest, CLIPResponse
from dinov2_pkg.srv import DINOv2, DINOv2Request, DINOv2Response
from dinov3_pkg.srv import DINOv3, DINOv3Request, DINOv3Response
from evit_sam_pkg.srv import (
    EvitSamSegmentation,
    EvitSamSegmentationRequest,
    EvitSamSegmentationResponse,
)
from yolo_world_pkg.srv import YoloDetection, YoloDetectionRequest
from ram_pkg.srv import RAMPlus, RAMPlusRequest
from coarse_localize.srv import (
    SingleExperiment,
    SingleExperimentRequest,
    SingleExperimentResponse,
)


class ForwardPipeline:
    def __init__(self):
        rospy.init_node("forward_pipeline")
        # ROS Params
        ## General
        self.cv_bridge = CvBridge()
        self.available_caod_heads = ["mvits", "ram"]
        self.available_seg_heads = ["fast-sam", "evit-sam"]
        self.available_feature_encode_heads = ["dinov2", "clip", "dinov3"]
        # Server & Client
        ## caod
        rospy.loginfo("Waiting for caod_pkg caod service...")
        rospy.wait_for_service("mvits")
        self.caod_client = rospy.ServiceProxy("mvits", CAOD)
        rospy.loginfo("Waiting for ram_pkg ram_plus service...")
        rospy.wait_for_service("ram_plus")
        self.ram_client = rospy.ServiceProxy("ram_plus", RAMPlus)
        ## sam_segmentation
        rospy.loginfo("Waiting for evit_sam_pkg sam_segmentation service...")
        rospy.wait_for_service("sam_segmentation")
        self.evit_sam_client = rospy.ServiceProxy(
            "sam_segmentation", EvitSamSegmentation
        )
        ## yolo_detection
        rospy.loginfo("Waiting for yolo_world_pkg yolo_detection service...")
        rospy.wait_for_service("yolo_detection")
        self.yolo_client = rospy.ServiceProxy("yolo_detection", YoloDetection)
        ## feature_encode
        rospy.loginfo("Waiting for clip_pkg clip service...")
        rospy.wait_for_service("clip")
        self.clip_client = rospy.ServiceProxy("clip", CLIP)
        rospy.loginfo("Waiting for dinov2_pkg dinov2 service...")
        rospy.wait_for_service("dinov2")
        self.dinov2_client = rospy.ServiceProxy("dinov2", DINOv2)
        rospy.loginfo("Waiting for dinov3_pkg dinov3 service...")
        rospy.wait_for_service("dinov3")
        self.dinov3_client = rospy.ServiceProxy("dinov3", DINOv3)
        # Publisher
        ## experiment
        rospy.Service(
            "~forward_pipeline", SingleExperiment, self._handle_single_experiment
        )
        rospy.loginfo("All required services are ready")
        rospy.loginfo(f"*" * 10 + " ForwardPipeline initialized " + "*" * 10)
        rospy.loginfo("Available CAOD Heads: {}".format(self.available_caod_heads))
        rospy.loginfo("Available SEG Heads: {}".format(self.available_seg_heads))
        rospy.loginfo(
            "Available Feature Encode Heads: {}".format(
                self.available_feature_encode_heads
            )
        )
        rospy.loginfo(f"*" * 10 + "*" * 30 + "*" * 10)

    def _gen_tag_ram(self, rgb_msg: Image):
        """Call ram_pkg RAM++ service"""
        ram_request = RAMPlusRequest()
        ram_request.color_image = rgb_msg
        ram_response = self.ram_client(ram_request)
        return ram_response

    def _obj_det_yolo(self, rgb_msg: Image, prompt: List[str]):
        """Call yolo-world open-vocab det service"""
        yolo_request = YoloDetectionRequest()
        yolo_request.color_image = rgb_msg
        yolo_request.prompt = [prompt] if type(prompt) is str else prompt
        yolo_response = self.yolo_client(yolo_request)
        return yolo_response

    def _obj_det_caod(
        self,
        rgb_msg: Image,
        mode: str = "nms",
        score_threshold: float = 0.1,
        nms_threshold: float = 0.5,
    ):
        """Call class-agnostic object detection service"""
        caod_request = CAODRequest()
        caod_request.color_image = rgb_msg
        caod_request.mode = mode
        caod_request.score_threshold = score_threshold
        caod_request.nms_threshold = nms_threshold
        caod_response = self.caod_client(caod_request)
        return caod_response

    def _seg_full(self, rgb_msg: Image):
        """Perform full image segmentation using EViT-SAM."""
        evit_request = EvitSamSegmentationRequest()
        evit_request.model = "fast-sam"
        evit_request.mode = "full"
        evit_request.color_image = rgb_msg
        evit_response = self.evit_sam_client(evit_request)
        return evit_response

    def _seg_bbox(
        self, rgb_msg: Image, boxes: List[Tuple[int, int, int, int]], model="evit-sam"
    ):
        """Perform bounding box segmentation using EViT-SAM."""
        evit_request = EvitSamSegmentationRequest()
        evit_request.model = model
        evit_request.mode = "bbox"
        evit_request.color_image = rgb_msg
        evit_request.boxes = boxes
        evit_response = self.evit_sam_client(evit_request)
        return evit_response

    def _encode_images_clip(self, rgb_blocks: List[np.ndarray]):
        """Extract CLIP features for RGB blocks."""
        clip_request = CLIPRequest()
        clip_request.mode = "encode_images"
        clip_request.images = [
            self.cv_bridge.cv2_to_imgmsg(block, "rgb8") for block in rgb_blocks
        ]
        clip_response = self.clip_client(clip_request)
        return clip_response

    def _encode_images_dinov2(self, rgb_blocks: List[np.ndarray]):
        """Extract DINOv2 features for RGB blocks."""
        dinov2_request = DINOv2Request()
        dinov2_request.mode = "encode_images"
        dinov2_request.images = [
            self.cv_bridge.cv2_to_imgmsg(block, "rgb8") for block in rgb_blocks
        ]
        dinov2_response = self.dinov2_client(dinov2_request)
        return dinov2_response

    def _encode_images_dinov3(self, rgb_blocks: List[np.ndarray]):
        """Extract DINOv3 features for RGB blocks."""
        dinov3_request = DINOv3Request()
        dinov3_request.mode = "encode_images"
        dinov3_request.images = [
            self.cv_bridge.cv2_to_imgmsg(block, "rgb8") for block in rgb_blocks
        ]
        dinov3_response = self.dinov3_client(dinov3_request)
        return dinov3_response

    def extract_bboxes(self, masks_np: List[np.ndarray]) -> List[tuple]:
        """Extract bounding boxes from segmentation masks."""
        bboxes = []
        for mask_np in masks_np:
            coords = np.argwhere(mask_np > 0)
            if coords.shape[0] == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bboxes.append((x_min, y_min, x_max, y_max))
        return bboxes

    def crop_rgb_blocks(
        self, rgb_image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
    ):
        """Crop RGB blocks based on bounding boxes."""
        blocks = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            block = rgb_image[y_min:y_max, x_min:x_max]
            blocks.append(block)
        return blocks

    def paint_mask(self, rgb_image: np.ndarray, masks: List[np.ndarray]):
        """Paint the segmentation masks on the RGB image."""
        for mask in masks:
            color = np.random.randint(0, 256, size=3).tolist()
            rgb_image[mask > 0] = color
        return rgb_image

    def compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray):
        """Compute the Intersection over Union (IoU) of two binary masks."""
        mask1_bin = mask1 > 0
        mask2_bin = mask2 > 0
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def merge_masks(
        self,
        masks_full_np: List[np.ndarray],
        masks_bbox_np: List[np.ndarray],
        iou_threshold: float = 0.5,
        contain_threshold: float = 0.8,
    ):
        """Merge segmentation masks from full and bounding box modes."""
        merged_masks = list(masks_bbox_np)  # reserve all bbox segmentation results
        for full_mask_np in masks_full_np:
            max_iou = 0.0
            contained = False
            for bbox_mask_np in masks_bbox_np:
                iou = self.compute_mask_iou(full_mask_np, bbox_mask_np)
                max_iou = max(max_iou, iou)
                intersection = np.logical_and(full_mask_np > 0, bbox_mask_np > 0).sum()
                full_area = (full_mask_np > 0).sum()
                if full_area > 0:
                    contain_ratio = intersection / full_area
                else:
                    contain_ratio = 0.0
                if contain_ratio > contain_threshold:
                    contained = True
                    break
            if max_iou < iou_threshold and not contained:
                merged_masks.append(full_mask_np)
        return merged_masks

    def _handle_single_experiment(self, req: SingleExperiment):
        """Handle a single experiment request."""
        time_dict = {"forward": time.time()}
        # Copy data
        time_dict["copy_data"] = time.time()
        rgb_msg = req.rgb_msg
        caod_head = req.caod_head
        seg_head = req.seg_head
        feature_encode_head = req.feature_encode_head
        if caod_head not in self.available_caod_heads:
            rospy.logerr(f"Unsupported caod_head: {caod_head}")
            return
        if seg_head not in self.available_seg_heads:
            rospy.logerr(f"Unsupported seg_head: {seg_head}")
            return
        if feature_encode_head not in self.available_feature_encode_heads:
            rospy.logerr(f"Unsupported feature_encode_head: {feature_encode_head}")
            return
        masks_merge_iou_threshold = req.masks_merge_iou_threshold
        masks_merge_contain_threshold = req.masks_merge_contain_threshold
        time_dict["copy_data"] = time.time() - time_dict["copy_data"]
        try:
            ################################Forward#########################################
            # 1. Perform segmentation-full
            time_dict["_seg_full"] = time.time()
            evit_response = self._seg_full(rgb_msg)
            seg_masks_full = evit_response.seg_masks  # List of ROS Image messages
            time_dict["_seg_full"] = time.time() - time_dict["_seg_full"]
            if not evit_response.seg_masks:
                rospy.logwarn("evit_response.seg_masks is empty.")
                res = SingleExperimentResponse()
                res.success = False
                res.message = "evit_response.seg_masks is empty."
                return res
            ################################################################################
            # 2. Perform caod detection
            ## CAOD Branch 1: mvits
            if caod_head == "mvits":
                time_dict["_obj_det_caod"] = time.time()
                det_response = self._obj_det_caod(rgb_msg, "nms", 0.1, 0.7)
                time_dict["_obj_det_caod"] = time.time() - time_dict["_obj_det_caod"]
            ## CAOD Branch 2: ram + yolo
            elif caod_head == "ram":
                time_dict["_gen_tag_ram"] = time.time()
                ram_response = self._gen_tag_ram(rgb_msg)
                time_dict["_gen_tag_ram"] = time.time() - time_dict["_gen_tag_ram"]
                time_dict["_obj_det_yolo"] = time.time()
                det_response = self._obj_det_yolo(rgb_msg, ram_response.en_list)
                time_dict["_obj_det_yolo"] = time.time() - time_dict["_obj_det_yolo"]
            if not det_response.boxes:
                rospy.logwarn("det_response.boxes is empty.")
                res = SingleExperimentResponse()
                res.success = False
                res.message = "det_response.boxes is empty."
                return res
            ################################################################################
            # 3. Perform segmentation-bbox
            time_dict["_seg_bbox"] = time.time()
            ## SEG Branch 1: evit-sam
            if seg_head == "evit-sam":
                evit_response = self._seg_bbox(
                    rgb_msg, det_response.boxes, model="evit-sam"
                )
            ## SEG Branch 2: fast-sam
            elif seg_head == "fast-sam":
                evit_response = self._seg_bbox(
                    rgb_msg, det_response.boxes, model="fast-sam"
                )
            ## SEG Branch 3: sam
            # elif seg_head == "sam":
            #     evit_response = self._seg_bbox(rgb_msg, det_response.boxes, model='sam')
            seg_masks_bbox = evit_response.seg_masks  # List of ROS Image messages
            time_dict["_seg_bbox"] = time.time() - time_dict["_seg_bbox"]
            if not evit_response.seg_masks:
                rospy.logwarn("evit_response.seg_masks is empty.")
                res = SingleExperimentResponse()
                res.success = False
                res.message = "evit_response.seg_masks is empty."
                return res
            ################################################################################
            # 4. Merge segmentation masks
            full_masks_np = [
                self.cv_bridge.imgmsg_to_cv2(mask_msg, "mono8")
                for mask_msg in seg_masks_full
            ]
            bbox_masks_np = [
                self.cv_bridge.imgmsg_to_cv2(mask_msg, "mono8")
                for mask_msg in seg_masks_bbox
            ]
            time_dict["merge_masks"] = time.time()
            seg_masks = self.merge_masks(
                full_masks_np,
                bbox_masks_np,
                masks_merge_iou_threshold,
                masks_merge_contain_threshold,
            )  # List of numpy arrays
            time_dict["merge_masks"] = time.time() - time_dict["merge_masks"]
            ################################################################################
            # 5. Extract bounding boxes and crop RGB blocks
            time_dict["extract_bboxes"] = time.time()
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            bboxes = self.extract_bboxes(seg_masks)
            rgb_blocks = self.crop_rgb_blocks(rgb_image, bboxes)
            time_dict["extract_bboxes"] = time.time() - time_dict["extract_bboxes"]
            ################################################################################
            # 6. Perform features extraction
            ## Feature Extraction Branch 1: CLIP
            if feature_encode_head == "clip":
                time_dict["_encode_images_clip"] = time.time()
                clip_response = self._encode_images_clip(rgb_blocks)
                time_dict["_encode_images_clip"] = (
                    time.time() - time_dict["_encode_images_clip"]
                )
                semantic_fts = clip_response.clip_fts
                semantic_ft_dim = clip_response.clip_ft_dim
            ## Feature Extraction Branch 2: DINOv2
            elif feature_encode_head == "dinov2":
                time_dict["_encode_images_dinov2"] = time.time()
                dinov2_response = self._encode_images_dinov2(rgb_blocks)
                time_dict["_encode_images_dinov2"] = (
                    time.time() - time_dict["_encode_images_dinov2"]
                )
                semantic_fts = dinov2_response.dinov2_fts
                semantic_ft_dim = dinov2_response.dinov2_ft_dim
            ## Feature Extraction Branch 3: DINOv3
            elif feature_encode_head == "dinov3":
                time_dict["_encode_images_dinov3"] = time.time()
                dinov2_response = self._encode_images_dinov3(rgb_blocks)
                time_dict["_encode_images_dinov3"] = (
                    time.time() - time_dict["_encode_images_dinov3"]
                )
                semantic_fts = dinov2_response.dinov3_fts
                semantic_ft_dim = dinov2_response.dinov3_ft_dim
            time_dict["forward"] = time.time() - time_dict["forward"]
            ################################################################################
            # Now we have:
            # seg_masks <List[np.ndarray]> : Merged segmentation masks
            # rgb_blocks <List[np.ndarray]> : Cropped RGB blocks of the seg_masks
            # semantic_fts <List[float]> : Extracted semantic features of the rgb_blocks
            # semantic_ft_dim <int> : Dimension of the semantic features
            ##############################evaluation########################################
            res = SingleExperimentResponse()
            res.success = True
            res.message = "Forward pipeline executed successfully."
            res.seg_masks = [
                self.cv_bridge.cv2_to_imgmsg(mask_np, "mono8") for mask_np in seg_masks
            ]
            res.semantic_fts = semantic_fts
            res.semantic_ft_dim = semantic_ft_dim
            res.time_forward = time_dict.get("forward", -1.0)
            res.time_copy_data = time_dict.get("copy_data", -1.0)
            res.time_seg_full = time_dict.get("_seg_full", -1.0)
            res.time_seg_bbox = time_dict.get("_seg_bbox", -1.0)
            res.time_obj_det_caod = time_dict.get("_obj_det_caod", -1.0)
            res.time_gen_tag_ram = time_dict.get("_gen_tag_ram", -1.0)
            res.time_obj_det_yolo = time_dict.get("_obj_det_yolo", -1.0)
            res.time_merge_masks = time_dict.get("merge_masks", -1.0)
            res.time_extract_bboxes = time_dict.get("extract_bboxes", -1.0)
            res.time_encode_images_clip = time_dict.get("_encode_images_clip", -1.0)
            res.time_encode_images_dinov2 = time_dict.get("_encode_images_dinov2", -1.0)
            rospy.loginfo(f"{'*'*10} ForwardPipeline Time Dict {'*'*10}")
            for key, val in time_dict.items():
                rospy.loginfo(f"{key}: {val:.3f} seconds")
            return res
        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")
            res = SingleExperimentResponse()
            res.success = False
            res.message = f"Processing error: {str(e)}"
            return res


if __name__ == "__main__":
    try:
        node = ForwardPipeline()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error in ForwardPipeline: {e}")
