import cv2
import numpy as np
import struct
import threading
import time
import traceback
from typing import List, Tuple

import rospy
import message_filters
import tf
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

from clip_pkg.srv import CLIP, CLIPRequest
from evit_sam_pkg.srv import EvitSamSegmentation, EvitSamSegmentationRequest
from yolo_world_pkg.srv import YoloDetection, YoloDetectionRequest
from fine_grained_search.srv import Query, QueryResponse
from pcd_utils import (
    convert_caminfo,
    convert_from_uvd,
    filter_points_dbscan,
    filter_valid_depth,
    random_sampling,
)
from preset_database import PresetDB


class FineGrainedSearchNode:
    def __init__(self):
        rospy.init_node("fine_grained_search_node")
        self.cv_bridge = CvBridge()

        # ROS Params
        ## General
        self.debug = rospy.get_param("~debug", True)
        ## Input - Camera Pose
        self.camera_link_father = rospy.get_param("~camera_link_father", "map")
        self.camera_link_child = rospy.get_param(
            "~camera_link_child", "camera_rgb_optical_frame"
        )
        ## Input - RGB and Depth
        self.rgb_sub_topic = rospy.get_param("~rgb_sub", "/ai2thor/rgb")
        self.depth_sub_topic = rospy.get_param("~depth_sub", "/ai2thor/depth")
        self.caminfo_sub_topic = rospy.get_param("~caminfo_sub", "/ai2thor/camera_info")
        ## Processing - Depth and Point Cloud
        self.depth_scale = rospy.get_param("~depth_scale", 1.0)
        self.depth_filter_min_depth = rospy.get_param(
            "~depth_filter_min_depth", 0.1
        )  # m
        self.depth_filter_max_depth = rospy.get_param("~depth_filter_max_depth", 10.0)
        self.enable_sample = rospy.get_param("~enable_sample", True)
        self.max_sample = rospy.get_param("~max_sample", 500)
        self.enable_filter = rospy.get_param("~enable_filter", True)
        self.enable_erode = rospy.get_param("~enable_erode", True)
        ## Processing - Yolo Det score threshold (Branch 0)
        self.yolo_det_score_threshold = rospy.get_param(
            "~yolo_det_score_threshold", 0.5
        )
        ## Processing - Masks Merge
        self.masks_merge_iou_threshold = rospy.get_param(
            "~masks_merge_iou_threshold", 0.5
        )
        self.masks_merge_contain_threshold = rospy.get_param(
            "~masks_merge_contain_threshold", 0.8
        )
        ## Processing - Preset DB
        preset_db_path = rospy.get_param("~preset_db_path", "preset.db")
        self.query_preset_threshold = rospy.get_param("~query_preset_threshold", 0.95)
        ## Output
        self.enable_show_annotated_frame = rospy.get_param(
            "~enable_show_annotated_frame", True
        )
        self.enable_show_fullseg_masks = rospy.get_param(
            "~enable_show_fullseg_masks", True
        )
        self.enable_show_merged_masks = rospy.get_param(
            "~enable_show_merged_masks", True
        )
        self.enable_show_query_pcd2 = rospy.get_param("~enable_show_query_pcd2", True)

        # Server & Client
        ## Query
        self.query_server = rospy.Service("~fine_query", Query, self._handle_query)
        ## sam_segmentation
        rospy.loginfo("Waiting for evit_sam_pkg sam_segmentation service...")
        rospy.wait_for_service("sam_segmentation")
        self.evit_sam_client = rospy.ServiceProxy(
            "sam_segmentation", EvitSamSegmentation
        )
        ## clip
        rospy.loginfo("Waiting for clip_pkg clip service...")
        rospy.wait_for_service("clip")
        self.clip_client = rospy.ServiceProxy("clip", CLIP)
        rospy.loginfo("All required services are ready")
        ## yolo
        rospy.loginfo("Waiting for yolo_world_pkg yolo_detection service...")
        rospy.wait_for_service("yolo_detection")
        self.yolo_client = rospy.ServiceProxy("yolo_detection", YoloDetection)

        # Data lock and state variables
        self.data_lock = threading.Lock()
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_caminfo_msg = None
        self.latest_pose = None

        # Initialize database
        self.preset_db = PresetDB(
            db_path=preset_db_path,
            renew_db=False,
        )

        # Subscriber & Publisher
        ## Subscribe to RGB, Depth, and Camera Info
        rgb_sub = message_filters.Subscriber(self.rgb_sub_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_sub_topic, Image)
        caminfo_sub = message_filters.Subscriber(self.caminfo_sub_topic, CameraInfo)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, caminfo_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self._sync_callback)
        ## Publishers
        self.query_pcd2_pub = rospy.Publisher("~query_pcd2", PointCloud2, queue_size=1)
        self.annotated_frame_pub = (
            rospy.Publisher("~annotated_frame", Image, queue_size=1)
            if self.enable_show_annotated_frame
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

        rospy.loginfo("FineGrainedSearchNode initialized.")

    def _sync_callback(self, rgb_msg: Image, depth_msg: Image, caminfo_msg: CameraInfo):
        """Sync callback for RGB, Depth, and CameraInfo messages"""
        try:
            transform = self.tf_buffer.lookup_transform(  # Lookup TF to get the pose
                self.camera_link_father,
                self.camera_link_child,
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
                self.latest_depth_msg = depth_msg
                self.latest_caminfo_msg = caminfo_msg
        except Exception as e:
            rospy.logwarn(f"TF lookup failed in sync_callback: {e}")
            with self.data_lock:
                self.latest_pose = np.eye(4)  # Use identity matrix as fallback
                self.latest_rgb_msg = rgb_msg
                self.latest_depth_msg = depth_msg
                self.latest_caminfo_msg = caminfo_msg

    def _obj_det_yolo(self, rgb_msg: Image, prompt: List[str]):
        """Call yolo-world open-vocab det service"""
        yolo_request = YoloDetectionRequest()
        yolo_request.color_image = rgb_msg
        yolo_request.prompt = [prompt] if type(prompt) is str else prompt
        yolo_response = self.yolo_client(yolo_request)
        return yolo_response

    def _seg_full(self, rgb_msg: Image):
        """Perform full image segmentation using EViT-SAM."""
        evit_request = EvitSamSegmentationRequest()
        evit_request.model = "fast"
        evit_request.mode = "full"
        evit_request.color_image = rgb_msg
        evit_response = self.evit_sam_client(evit_request)
        return evit_response

    def _seg_bbox(self, rgb_msg: Image, boxes: List[Tuple[int, int, int, int]]):
        """Perform bounding box segmentation using EViT-SAM."""
        evit_request = EvitSamSegmentationRequest()
        evit_request.model = "evit"
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

    def _encode_text_clip(self, query_text: str):
        """Encode text using CLIP."""
        req = CLIPRequest()
        req.mode = "encode_text"
        req.text = query_text
        res = self.clip_client(req)
        return res.clip_fts

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

    def create_pcd2(self, points_3d, colors):
        """Create a PointCloud2 message from 3D points and colors."""
        n_points = points_3d.shape[0]
        data = []
        for i in range(n_points):
            rgb_packed = (
                (int(colors[i, 0]) << 16) | (int(colors[i, 1]) << 8) | int(colors[i, 2])
            )
            data.append(
                struct.pack(
                    "fffI",
                    points_3d[i, 0],
                    points_3d[i, 1],
                    points_3d[i, 2],
                    rgb_packed,
                )
            )
        cloud = PointCloud2()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = self.camera_link_father
        cloud.height = 1
        cloud.width = n_points
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * n_points
        cloud.is_dense = True
        cloud.data = b"".join(data)
        return cloud

    def create_pcd_np(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        seg_mask: np.ndarray,
        valid_depth_image_mask: np.ndarray,
        intrinsic: np.ndarray,
        pose: np.ndarray,
        erode=True,
        sample=True,
        filter=True,
    ):
        """Create a point cloud from RGB-D images and segmentation masks."""
        pcd_mask = (seg_mask > 0) & (valid_depth_image_mask > 0)
        if erode:
            kernel = np.ones((3, 3), np.uint8)
            pcd_mask = cv2.erode(
                pcd_mask.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        v_coords, u_coords = np.where(pcd_mask)
        z_values = depth_image[v_coords, u_coords]
        colors = rgb_image[v_coords, u_coords]
        points_np = convert_from_uvd(  # np.array - shape: (N, 3)
            u_coords, v_coords, z_values, intrinsic, pose, depth_scale=self.depth_scale
        )
        if filter and len(points_np) > 0:
            points_np, colors = filter_points_dbscan(points_np, colors)
        if sample and len(points_np) > self.max_sample:
            indices = np.random.choice(len(points_np), self.max_sample, replace=False)
            points_np = points_np[indices]
            colors = colors[indices]
        return points_np, colors

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

    def find_best_match_idx(self, masks_fts: np.ndarray, query_ft: np.ndarray):
        """Find the best matching mask for the given query feature."""
        similarities = np.dot(masks_fts, query_ft.T)
        best_match_idx = np.argmax(similarities, axis=0)
        best_match_idx = (
            best_match_idx.item()
            if isinstance(best_match_idx, np.ndarray)
            else best_match_idx
        )
        return best_match_idx

    def calc_pcd_np_aabb_min_max_center(self, points_np: np.ndarray):
        """Calculate the AABB (Axis-Aligned Bounding Box) min, max from point cloud data."""
        if points_np.size == 0:
            return None, None, None
        min_xyz = np.min(points_np, axis=0)  # [x_min, y_min, z_min]
        max_xyz = np.max(points_np, axis=0)  # [x_max, y_max, z_max]
        center = (min_xyz + max_xyz) / 2.0  # [x_center, y_center, z_center]
        return min_xyz, max_xyz, center

    def _handle_query(self, req: Query):
        """Handle query requests."""
        query = req.query
        res = QueryResponse()
        time_dict = {"total": time.time()}
        # Copy data
        time_dict["copy_data"] = time.time()
        with self.data_lock:
            if (
                self.latest_rgb_msg is None
                or self.latest_depth_msg is None
                or self.latest_caminfo_msg is None
            ):
                return
            rgb_msg = self.latest_rgb_msg
            depth_msg = self.latest_depth_msg
            caminfo_msg = self.latest_caminfo_msg
            pose = self.latest_pose.copy()
            self.latest_rgb_msg = None
            self.latest_depth_msg = None
            self.latest_caminfo_msg = None
            self.latest_pose = None
        time_dict["copy_data"] = time.time() - time_dict["copy_data"]
        branch = [False, False]
        try:
            # ROS msg to OpenCV
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            camera_info = convert_caminfo(caminfo_msg)

            # Perform yolo detection
            time_dict["_obj_det_yolo"] = time.time()
            det_response = self._obj_det_yolo(rgb_msg, [query])
            time_dict["_obj_det_yolo"] = time.time() - time_dict["_obj_det_yolo"]

            # Perform segmentation-bbox
            if det_response.boxes:
                time_dict["_seg_bbox"] = time.time()
                evit_response = self._seg_bbox(rgb_msg, det_response.boxes)
                seg_masks_bbox = evit_response.seg_masks  # List of ROS Image messages
                time_dict["_seg_bbox"] = time.time() - time_dict["_seg_bbox"]
            else:
                seg_masks_bbox = []

            # Branch 0: Object Detected with high confidence
            if (
                det_response.scores
                and max(det_response.scores) >= self.yolo_det_score_threshold
            ):
                branch[0] = True
                # Find best match mask
                time_dict["find_best_match"] = time.time()
                det_scores_np = np.array(det_response.scores)
                best_match_idx = np.argmax(det_scores_np)
                best_match_mask = seg_masks_bbox[best_match_idx]
                best_match_mask = self.cv_bridge.imgmsg_to_cv2(best_match_mask, "mono8")
                time_dict["find_best_match"] = (
                    time.time() - time_dict["find_best_match"]
                )
            # Branch 1: Object Not Detected or Low Confidence
            else:
                branch[1] = True
                # Perform segmentation-full
                time_dict["_seg_full"] = time.time()
                evit_response = self._seg_full(rgb_msg)
                seg_masks_full = evit_response.seg_masks  # List of ROS Image messages
                time_dict["_seg_full"] = time.time() - time_dict["_seg_full"]

                # Merge segmentation masks
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

                # Extract bounding boxes and crop RGB blocks
                time_dict["extract_bboxes"] = time.time()
                bboxes = self.extract_bboxes(seg_masks)
                rgb_blocks = self.crop_rgb_blocks(rgb_image, bboxes)
                time_dict["extract_bboxes"] = time.time() - time_dict["extract_bboxes"]
                assert len(rgb_blocks) == len(
                    seg_masks
                ), "Mismatch at len(rgb_blocks) == len(seg_masks)"

                # Extract masks CLIP features
                time_dict["_encode_images_clip"] = time.time()
                clip_response = self._encode_images_clip(rgb_blocks)
                masks_fts = np.array(clip_response.clip_fts).reshape(
                    -1, clip_response.clip_ft_dim
                )
                time_dict["_encode_images_clip"] = (
                    time.time() - time_dict["_encode_images_clip"]
                )
                assert len(masks_fts) == len(
                    seg_masks
                ), "Mismatch at len(masks_fts) == len(seg_masks)"

                # Extract query text
                time_dict["_encode_text_clip"] = time.time()
                query_ft = self._encode_text_clip(query)  # List[float]
                time_dict["_encode_text_clip"] = (
                    time.time() - time_dict["_encode_text_clip"]
                )

                # query preset db first
                time_dict["query_preset_db"] = time.time()
                preset_res = self.preset_db.query_by_clip_text_ft(query_ft, 1)
                if (
                    preset_res
                    and preset_res[0]["similarity"] >= self.query_preset_threshold
                ):
                    query_ft = preset_res[0][
                        "clip_ft_img"
                    ]  # replace query_ft with preset image feature
                    rospy.logwarn(
                        f"Preset query result: {preset_res[0]['text']}, {preset_res[0]['similarity']:.3f}. Using preset_img_ft as query_tf."
                    )
                time_dict["query_preset_db"] = (
                    time.time() - time_dict["query_preset_db"]
                )

                # Find best match mask
                time_dict["find_best_match"] = time.time()
                best_match_idx = self.find_best_match_idx(masks_fts, np.array(query_ft))
                best_match_mask = seg_masks[best_match_idx]
                time_dict["find_best_match"] = (
                    time.time() - time_dict["find_best_match"]
                )

            # Get the best matching mask valid depth
            time_dict["filter_valid_depth"] = time.time()
            valid_depth_mask = filter_valid_depth(
                depth_image,
                self.depth_filter_min_depth,
                self.depth_filter_max_depth,
                self.depth_scale,
            )
            time_dict["filter_valid_depth"] = (
                time.time() - time_dict["filter_valid_depth"]
            )

            # Create best match PCD
            time_dict["create_pcd_np"] = time.time()
            best_match_points_np, best_match_colors = self.create_pcd_np(
                rgb_image,
                depth_image,
                best_match_mask,
                valid_depth_mask,
                camera_info,
                pose,
                self.enable_erode,
                self.enable_sample,
                self.enable_filter,
            )
            time_dict["create_pcd_np"] = time.time() - time_dict["create_pcd_np"]

            # DEBUG
            time_dict["total"] = time.time() - time_dict["total"]
            if self.debug:
                rospy.loginfo(f"{'*' * 10} Memory Generator Debug Info {'*' * 10}")
                rospy.loginfo(f"Query: {query}, Branch: {branch}")
                for key, val in time_dict.items():
                    rospy.loginfo(f"{key}: {val:.3f} seconds")
                if self.enable_show_query_pcd2:
                    self.query_pcd2_pub.publish(
                        self.create_pcd2(best_match_points_np, best_match_colors)
                    )
                if self.enable_show_annotated_frame:
                    self.annotated_frame_pub.publish(det_response.annotated_frame)
                if self.enable_show_fullseg_masks and branch[1]:
                    fullseg_image = rgb_image.copy()
                    painted_fullseg_image = self.paint_mask(
                        fullseg_image, full_masks_np
                    )
                    self.fullseg_masks_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(painted_fullseg_image, "rgb8")
                    )
                if self.enable_show_merged_masks and branch[1]:
                    merged_image = rgb_image.copy()
                    painted_merged_image = self.paint_mask(merged_image, seg_masks)
                    self.merged_masks_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(painted_merged_image, "rgb8")
                    )

            # Create response
            res.success = True
            res.message = f"Query: {query}, Branch: {branch}"
            aabb_min, aabb_max, center = self.calc_pcd_np_aabb_min_max_center(
                best_match_points_np
            )
            res.aabb_min = aabb_min.tolist() if aabb_min is not None else []
            res.aabb_max = aabb_max.tolist() if aabb_max is not None else []
            res.center = center.tolist() if center is not None else []
            return res

        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")


if __name__ == "__main__":
    try:
        node = FineGrainedSearchNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error in Fine Grained Search Node: {e}")
