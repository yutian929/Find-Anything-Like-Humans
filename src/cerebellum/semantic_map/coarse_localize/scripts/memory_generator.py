import threading
import time
import numpy as np
from typing import List, Tuple, Union

import message_filters
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from coarse_localize.msg import Memory
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


class MemoryGenerator:
    def __init__(self):
        rospy.init_node("memory_generator")
        # ROS Params
        ## General
        self.debug = rospy.get_param("~debug", True)
        self.main_loop_duration = rospy.get_param("~main_loop_duration", 0.1)
        self.cv_bridge = CvBridge()
        self.caod_head = rospy.get_param("~caod_head", "mvits")  # mvits / ram
        self.seg_head = rospy.get_param(
            "~seg_head", "evit-sam"
        )  # evit-sam / fast-sam / sam
        self.feature_encode_head = rospy.get_param(
            "~feature_encode_head", "dinov3"
        )  # clip / dinov2 / dinov3
        ## Input - Pose
        self.pose_link_father = rospy.get_param("~pose_link_father", "map")
        self.pose_link_child = rospy.get_param("~pose_link_child", "camera_link")
        ## Input - RGB
        self.rgb_sub_topic = rospy.get_param("~rgb_sub", "/ai2thor/rgb")
        ## Processing - Masks Merge
        self.masks_merge_iou_threshold = rospy.get_param(
            "~masks_merge_iou_threshold", 0.5
        )
        self.masks_merge_contain_threshold = rospy.get_param(
            "~masks_merge_contain_threshold", 0.8
        )
        ## Output
        self.enable_pub_memory = rospy.get_param("~enable_pub_memory", True)
        self.enable_show_annotated_frame = rospy.get_param(
            "~enable_show_annotated_frame", True
        )
        self.enable_show_det_masks = rospy.get_param("~enable_show_det_masks", True)
        self.enable_show_fullseg_masks = rospy.get_param(
            "~enable_show_fullseg_masks", True
        )
        self.enable_show_merged_masks = rospy.get_param(
            "~enable_show_merged_masks", True
        )
        # Server & Client
        ## caod
        if self.caod_head == "mvits":
            rospy.loginfo("Waiting for caod_pkg caod service...")
            rospy.wait_for_service("mvits")
            self.caod_client = rospy.ServiceProxy("mvits", CAOD)
        elif self.caod_head == "ram":
            rospy.loginfo("Waiting for ram_pkg ram_plus service...")
            rospy.wait_for_service("ram_plus")
            self.ram_client = rospy.ServiceProxy("ram_plus", RAMPlus)
        else:
            rospy.logerr("Unsupportable caod_head !")
            return
        ## sam_segmentation
        rospy.loginfo("Waiting for evit_sam_pkg sam_segmentation service...")
        rospy.wait_for_service("sam_segmentation")
        self.evit_sam_client = rospy.ServiceProxy(
            "sam_segmentation", EvitSamSegmentation
        )
        ## feature_encode
        if self.feature_encode_head == "clip":
            rospy.loginfo("Waiting for clip_pkg clip service...")
            rospy.wait_for_service("clip")
            self.clip_client = rospy.ServiceProxy("clip", CLIP)
        elif self.feature_encode_head == "dinov2":
            rospy.loginfo("Waiting for dinov2_pkg dinov2 service...")
            rospy.wait_for_service("dinov2")
            self.dinov2_client = rospy.ServiceProxy("dinov2", DINOv2)
        elif self.feature_encode_head == "dinov3":
            rospy.loginfo("Waiting for dinov3_pkg dinov3 service...")
            rospy.wait_for_service("dinov3")
            self.dinov3_client = rospy.ServiceProxy("dinov3", DINOv3)
        else:
            rospy.logerr("Unsupportable feature_encode_head !")
            return
        ## yolo
        rospy.loginfo("Waiting for yolo_world_pkg yolo_detection service...")
        rospy.wait_for_service("yolo_detection")
        self.yolo_client = rospy.ServiceProxy("yolo_detection", YoloDetection)
        rospy.loginfo("All required services are ready")

        # Data lock and state variables
        self.data_lock = threading.Lock()
        self.latest_rgb_msg = None
        self.latest_pose = None

        # Subscriber & Publisher
        ## Subscribe to RGB and synchronize with TF
        rgb_sub = message_filters.Subscriber(self.rgb_sub_topic, Image)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self._sync_callback)
        ## Publisher
        self.memory_pub = rospy.Publisher("~memory", Memory, queue_size=1)
        self.annotated_frame_pub = (
            rospy.Publisher("~annotated_frame", Image, queue_size=1)
            if self.enable_show_annotated_frame
            else None
        )
        self.det_masks_pub = (
            rospy.Publisher("~det_masks", Image, queue_size=1)
            if self.enable_show_det_masks
            else None
        )
        self.fullseg_masks_pub = (
            rospy.Publisher("~fullseg_masks", Image, queue_size=1)
            if self.enable_show_fullseg_masks
            else None
        )
        self.merged_masks_pub = (
            rospy.Publisher("~merged_masks", Image, queue_size=1)
            if self.enable_show_merged_masks
            else None
        )

        # Main loop
        rospy.Timer(rospy.Duration(self.main_loop_duration), self._timer_callback)
        rospy.loginfo(f"*" * 10 + " Memory Generator initialized " + "*" * 10)
        rospy.loginfo("CAOD Head: {}".format(self.caod_head))
        rospy.loginfo("SEG Head: {}".format(self.seg_head))
        rospy.loginfo("Feature Encode Head: {}".format(self.feature_encode_head))
        rospy.loginfo(f"*" * 10 + "*" * 30 + "*" * 10)

    def _sync_callback(self, rgb_msg: Image):
        """Synchronize RGB image and pose."""
        try:
            transform = self.tf_buffer.lookup_transform(  # Lookup TF to get the pose
                self.pose_link_father,
                self.pose_link_child,
                rgb_msg.header.stamp,
                rospy.Duration(0.1),
            )
            t = transform.transform.translation
            q = transform.transform.rotation
            T = tf.transformations.quaternion_matrix(
                [q.x, q.y, q.z, q.w]
            )  # Convert to 4x4 matrix
            T[0, 3] = t.x
            T[1, 3] = t.y
            T[2, 3] = t.z
            with self.data_lock:
                self.latest_pose = T
                self.latest_rgb_msg = rgb_msg
        except Exception as e:
            rospy.logwarn(f"TF lookup failed in sync_callback: {e}")
            with self.data_lock:
                self.latest_pose = np.eye(4)  # Use identity matrix as fallback
                self.latest_rgb_msg = rgb_msg

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

    def _pub_memory_msg(self, pose_params: List[float], semantic_fts, semantic_ft_dim):
        """Create and publish a Memory message."""
        memory_msg = Memory()
        memory_msg.header = Header(
            stamp=rospy.Time.now(), frame_id=self.pose_link_father
        )
        memory_msg.pose_params = pose_params
        memory_msg.semantic_fts = semantic_fts
        memory_msg.semantic_ft_dim = semantic_ft_dim
        self.memory_pub.publish(memory_msg)

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

    def _timer_callback(self, event):
        """Periodic processing callback. Main Loop"""
        time_dict = {"total": time.time()}
        # Copy data
        time_dict["copy_data"] = time.time()
        with self.data_lock:
            if self.latest_rgb_msg is None:
                return
            rgb_msg = self.latest_rgb_msg
            pose = self.latest_pose.copy()
            self.latest_rgb_msg = None
            self.latest_pose = None
        time_dict["copy_data"] = time.time() - time_dict["copy_data"]
        try:
            ####################### segmentation-full ########################
            time_dict["_seg_full"] = time.time()
            evit_response = self._seg_full(rgb_msg)
            seg_masks_full = evit_response.seg_masks  # List of ROS Image messages
            time_dict["_seg_full"] = time.time() - time_dict["_seg_full"]
            if not evit_response.seg_masks:
                rospy.logwarn("evit_response.seg_masks is empty.")
                return
            ###################### caod detection ############################
            ## CAOD Branch 1: mvits
            if self.caod_head == "mvits":
                time_dict["_obj_det_caod"] = time.time()
                det_response = self._obj_det_caod(rgb_msg, "nms", 0.1, 0.7)
                time_dict["_obj_det_caod"] = time.time() - time_dict["_obj_det_caod"]
            ## CAOD Branch 2: ram + yolo
            elif self.caod_head == "ram":
                time_dict["_gen_tag_ram"] = time.time()
                ram_response = self._gen_tag_ram(rgb_msg)
                time_dict["_gen_tag_ram"] = time.time() - time_dict["_gen_tag_ram"]
                time_dict["_obj_det_yolo"] = time.time()
                det_response = self._obj_det_yolo(rgb_msg, ram_response.en_list)
                time_dict["_obj_det_yolo"] = time.time() - time_dict["_obj_det_yolo"]
            if not det_response.boxes:
                rospy.logwarn("det_response.boxes is empty.")
                return
            ##################### segmentation-bbox ##########################
            time_dict["_seg_bbox"] = time.time()
            ## SEG Branch 1: evit-sam
            if self.seg_head == "evit-sam":
                evit_response = self._seg_bbox(
                    rgb_msg, det_response.boxes, model="evit-sam"
                )
            ## SEG Branch 2: fast-sam
            elif self.seg_head == "fast-sam":
                evit_response = self._seg_bbox(
                    rgb_msg, det_response.boxes, model="fast-sam"
                )
            ## SEG Branch 3: sam
            elif self.seg_head == "sam":
                evit_response = self._seg_bbox(rgb_msg, det_response.boxes, model="sam")
            seg_masks_bbox = evit_response.seg_masks  # List of ROS Image messages
            time_dict["_seg_bbox"] = time.time() - time_dict["_seg_bbox"]
            if not evit_response.seg_masks:
                rospy.logwarn("evit_response.seg_masks is empty.")
                return
            ################# Merge segmentation masks ######################
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
                self.masks_merge_iou_threshold,
                self.masks_merge_contain_threshold,
            )  # List of numpy arrays
            time_dict["merge_masks"] = time.time() - time_dict["merge_masks"]
            ######### Extract bounding boxes and crop RGB blocks ############
            time_dict["extract_bboxes"] = time.time()
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            bboxes = self.extract_bboxes(seg_masks)
            rgb_blocks = self.crop_rgb_blocks(rgb_image, bboxes)
            time_dict["extract_bboxes"] = time.time() - time_dict["extract_bboxes"]
            ############### features extraction ###############################
            ## Feature Extraction Branch 1: CLIP
            if self.feature_encode_head == "clip":
                time_dict["_encode_images_clip"] = time.time()
                fts_extract_response = self._encode_images_clip(rgb_blocks)
                semantic_fts = fts_extract_response.clip_fts
                semantic_ft_dim = fts_extract_response.clip_ft_dim
                time_dict["_encode_images_clip"] = (
                    time.time() - time_dict["_encode_images_clip"]
                )
            ## Feature Extraction Branch 2: DINOv2
            elif self.feature_encode_head == "dinov2":
                time_dict["_encode_images_dinov2"] = time.time()
                fts_extract_response = self._encode_images_dinov2(rgb_blocks)
                semantic_fts = fts_extract_response.dinov2_fts
                semantic_ft_dim = fts_extract_response.dinov2_ft_dim
                time_dict["_encode_images_dinov2"] = (
                    time.time() - time_dict["_encode_images_dinov2"]
                )
            ## Feature Extraction Branch 3: DINOv3
            elif self.feature_encode_head == "dinov3":
                time_dict["_encode_images_dinov3"] = time.time()
                fts_extract_response = self._encode_images_dinov3(rgb_blocks)
                semantic_fts = fts_extract_response.dinov3_fts
                semantic_ft_dim = fts_extract_response.dinov3_ft_dim
                time_dict["_encode_images_dinov3"] = (
                    time.time() - time_dict["_encode_images_dinov3"]
                )
            ################### Publish Memory message ######################
            if self.enable_pub_memory:
                time_dict["_pub_memory_msg"] = time.time()
                x = pose[0, 3]
                y = pose[1, 3]
                z = pose[2, 3]
                qx, qy, qz, qw = tf.transformations.quaternion_from_euler(
                    *tf.transformations.euler_from_matrix(pose)
                )
                self._pub_memory_msg(
                    [x, y, z, qx, qy, qz, qw], semantic_fts, semantic_ft_dim
                )
                time_dict["_pub_memory_msg"] = (
                    time.time() - time_dict["_pub_memory_msg"]
                )
            ########################## DEBUG ###############################
            time_dict["total"] = time.time() - time_dict["total"]
            if self.debug:
                rospy.loginfo(f"{'*' * 10} Memory Generator Debug Info {'*' * 10}")
                for key, val in time_dict.items():
                    rospy.loginfo(f"{key}: {val:.3f} seconds")
                if self.enable_show_det_masks:
                    det_image = rgb_image.copy()
                    painted_det_image = self.paint_mask(det_image, bbox_masks_np)
                    self.det_masks_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(painted_det_image, "rgb8")
                    )
                if self.enable_show_fullseg_masks:
                    fullseg_image = rgb_image.copy()
                    painted_fullseg_image = self.paint_mask(
                        fullseg_image, full_masks_np
                    )
                    self.fullseg_masks_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(painted_fullseg_image, "rgb8")
                    )
                if self.enable_show_merged_masks:
                    merged_image = rgb_image.copy()
                    painted_merged_image = self.paint_mask(merged_image, seg_masks)
                    self.merged_masks_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(painted_merged_image, "rgb8")
                    )
                if self.enable_show_annotated_frame:
                    self.annotated_frame_pub.publish(det_response.annotated_frame)
        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")


if __name__ == "__main__":
    try:
        node = MemoryGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error in Memory Generator: {e}")
