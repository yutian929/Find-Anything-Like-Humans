#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import cv2
import rospy
import tqdm
from typing import List
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from clip_pkg.srv import CLIP, CLIPRequest, CLIPResponse
from dinov2_pkg.srv import DINOv2, DINOv2Request, DINOv2Response
from dinov3_pkg.srv import DINOv3, DINOv3Request, DINOv3Response
from coarse_localize.srv import (
    SingleExperiment,
    SingleExperimentRequest,
    SingleExperimentResponse,
)
from scene_memory_metrics import (
    SceneMemoryMetrics,
    MetricsAggregator,
    fps_from_times,
    make_id2name,
)


class Evaluation:
    def __init__(self):
        rospy.init_node("evaluation")
        # ROS Params
        ## General
        self.scannet_root = rospy.get_param(
            "~scannet_root", "/home/yutian/下载/scannetv2"
        )
        self.scannet_scene = rospy.get_param("~scannet_scene", "scene0347_00")
        self.scene_dir = os.path.join(self.scannet_root, self.scannet_scene)
        # self.eval_caod_heads = ["mvits", "ram"]
        # self.eval_seg_heads = ["fast-sam", "evit-sam"]
        # self.eval_feature_encode_heads = ["clip", "dinov2", "dinov3"]
        self.eval_caod_heads = ["ram"]
        self.eval_seg_heads = ["evit-sam"]
        self.eval_feature_encode_heads = ["dinov3"]
        self.cv_bridge = CvBridge()
        self.masks_merge_iou_threshold = rospy.get_param(
            "~masks_merge_iou_threshold", 0.5
        )
        self.masks_merge_contain_threshold = rospy.get_param(
            "~masks_merge_contain_threshold", 0.8
        )
        if not self.scene_dir or not os.path.exists(self.scene_dir):
            rospy.logerr("Invalid ~scene_dir: %s", self.scene_dir)
            return
        # Server & Client
        ## text_encode
        rospy.loginfo("Waiting for clip_pkg clip service...")
        rospy.wait_for_service("clip")
        self.clip_client = rospy.ServiceProxy("clip", CLIP)
        rospy.loginfo("Waiting for dinov2_pkg dinov2 service...")
        rospy.wait_for_service("dinov2")
        self.dinov2_client = rospy.ServiceProxy("dinov2", DINOv2)
        rospy.loginfo("Waiting for dinov3_pkg dinov3 service...")
        rospy.wait_for_service("dinov3")
        self.dinov3_client = rospy.ServiceProxy("dinov3", DINOv3)
        rospy.loginfo("Waiting for coarse_localize_pkg forward_pipeline service...")
        ## forward_pipeline
        rospy.wait_for_service("/forward_pipeline/forward_pipeline")
        self.forward_pipeline_client = rospy.ServiceProxy(
            "/forward_pipeline/forward_pipeline", SingleExperiment
        )
        # Publisher
        self.origin_rgb_pub = rospy.Publisher("~origin_rgb", Image, queue_size=1)
        self.forward_masks_pub = rospy.Publisher("~forward_masks", Image, queue_size=1)
        self.ground_instance_masks_pub = rospy.Publisher(
            "~ground_instance_masks", Image, queue_size=1
        )
        self.ground_label_masks_pub = rospy.Publisher(
            "~ground_label_masks", Image, queue_size=1
        )
        # Metrics
        self.show_vis_img = rospy.get_param("~show_vis_img", True)
        self.smm = SceneMemoryMetrics(
            seg_iou_thr=0.5,
            retr_cov_thr=0.5,  # 按需可设 0.3/0.4/0.5
            retrieval_ks=(1, 3),
            prompt_prefix="a photo of a ",
        )
        self.metrics_by_combo = {}
        self.forward_times_by_combo = {}
        self.ros_distri_times_by_combo = {}

    def _read_scannet(self):
        """Read ScanNetV2 dataset with intersection of color/instance/label frames."""
        id_category_json = os.path.join(self.scannet_root, "scannetv2_id_category.json")
        entries = {
            "entry_num": 0,
            "scene_dir": self.scene_dir,
            "id_category_json": id_category_json,
            "color_files": [],
            "2d-instance-filt_files": [],
            "2d-label-filt_files": [],
        }

        color_dir = os.path.join(self.scene_dir, "color")
        instance_dir = os.path.join(
            self.scene_dir, f"{self.scannet_scene}_2d-instance-filt/instance-filt"
        )
        label_dir = os.path.join(
            self.scene_dir, f"{self.scannet_scene}_2d-label-filt/label-filt"
        )
        for d in [color_dir, instance_dir, label_dir]:
            if not os.path.exists(d):
                rospy.logerr("Directory does not exist: %s", d)
                return None

        rgb_files = {
            os.path.splitext(f)[0]: os.path.join(color_dir, f)
            for f in os.listdir(color_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        inst_files = {
            os.path.splitext(f)[0]: os.path.join(instance_dir, f)
            for f in os.listdir(instance_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        label_files = {
            os.path.splitext(f)[0]: os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        common_keys = sorted(
            set(rgb_files.keys()) & set(inst_files.keys()) & set(label_files.keys())
        )
        if not common_keys:
            rospy.logwarn("No common frames found between RGB / instance / label.")
            return None

        entries["color_files"] = [rgb_files[k] for k in common_keys]
        entries["2d-instance-filt_files"] = [inst_files[k] for k in common_keys]
        entries["2d-label-filt_files"] = [label_files[k] for k in common_keys]
        entries["entry_num"] = len(common_keys)

        return entries

    def _encode_text_clip(self, query_text: str):
        """Encode text using CLIP."""
        clip_request = CLIPRequest()
        clip_request.mode = "encode_text"
        clip_request.text = query_text
        clip_response = self.clip_client(clip_request)
        return clip_response

    def _encode_text_dinov2(self, query_text: str):
        """Encode text using DINOv2."""
        dinov2_request = DINOv2Request()
        dinov2_request.mode = "encode_text"
        dinov2_request.text = query_text
        dinov2_response = self.dinov2_client(dinov2_request)
        return dinov2_response

    def _encode_text_dinov3(self, query_text: str):
        """Encode text using DINOv3."""
        dinov3_request = DINOv3Request()
        dinov3_request.mode = "encode_text"
        dinov3_request.text = query_text
        dinov3_response = self.dinov3_client(dinov3_request)
        return dinov3_response

    def paint_mask(self, rgb_image: np.ndarray, masks: List[np.ndarray]):
        """Paint the segmentation masks on the RGB image."""
        for mask in masks:
            color = np.random.randint(0, 256, size=3).tolist()
            rgb_image[mask > 0] = color
        return rgb_image

    def _visualize(
        self,
        rgb: np.ndarray,
        seg_masks_np: np.ndarray,  # (N,H,W) mono8/0-255
        inst_png: np.ndarray,  # (H,W) uint8
        label_png: np.ndarray,
    ):  # (H,W) uint16
        """简单可视化：原图、预测融合掩码、GT instance、GT label（不标 ID）"""

        # 1) 原图
        self.origin_rgb_pub.publish(self.cv_bridge.cv2_to_imgmsg(rgb, encoding="rgb8"))

        # 一个固定的配色器：同一次调用内稳定可复现；如需跨帧稳定可把种子挪到 __init__
        rng = np.random.RandomState(42)

        # 2) 预测融合掩码（seg_masks_np: (N,H,W)）
        pred_overlay = rgb.copy()
        if isinstance(seg_masks_np, np.ndarray) and seg_masks_np.ndim == 3:
            for i in range(seg_masks_np.shape[0]):
                m = seg_masks_np[i]
                if m.ndim != 2:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                color = rng.randint(0, 256, size=3, dtype=np.uint8)
                pred_overlay[m > 0] = color
        else:
            rospy.logwarn("seg_masks_np shape is not (N,H,W); skip pred overlay.")
        self.forward_masks_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(pred_overlay, "rgb8")
        )

        # 3) GT instance（inst_png: uint8, 0=bg, >0=实例ID）
        if inst_png is not None:
            inst_vis = rgb.copy()
            uniq = np.unique(inst_png)
            for iid in uniq:
                if iid == 0:
                    continue
                color = rng.randint(0, 256, size=3, dtype=np.uint8)
                inst_vis[inst_png == iid] = color
            self.ground_instance_masks_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(inst_vis, "rgb8")
            )

        # 4) GT label（label_png: uint16, 0=bg, >0=类ID）
        if label_png is not None:
            lab_vis = rgb.copy()
            uniq = np.unique(label_png)
            for cid in uniq:
                if cid == 0:
                    continue
                color = rng.randint(0, 256, size=3, dtype=np.uint8)
                lab_vis[label_png == cid] = color
            self.ground_label_masks_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(lab_vis, "rgb8")
            )

    def _text_encoder_for_feature(self, feature_encode_head: str):
        def enc_clip(txt: str):
            resp = self._encode_text_clip(txt)
            return np.array(resp.clip_fts)

        def enc_dinov2(txt: str):
            resp = self._encode_text_dinov2(txt)
            return np.array(resp.dinov2_fts)

        def enc_dinov3(txt: str):
            resp = self._encode_text_dinov3(txt)
            return np.array(resp.dinov3_fts)

        if feature_encode_head not in ["clip", "dinov2", "dinov3"]:
            rospy.logfatal(
                "Unknown feature_encode_head: %s, default to clip", feature_encode_head
            )
        return {"clip": enc_clip, "dinov2": enc_dinov2, "dinov3": enc_dinov3}.get(
            feature_encode_head, enc_clip
        )

    def run_eval(self):
        # Read dataset
        entries = self._read_scannet()
        if entries is None or entries["entry_num"] == 0:
            rospy.logerr("No valid entries found in dataset.")
            return
        with open(entries["id_category_json"], "r") as f:
            id_category = json.load(f)
        if id_category is None or len(id_category) == 0:
            rospy.logerr("Failed to load id_category_json or it's empty.")
            return
        # Arrange all configuration combinations
        for caod_head in self.eval_caod_heads:
            for seg_head in self.eval_seg_heads:
                for feature_encode_head in self.eval_feature_encode_heads:
                    for idx in tqdm.tqdm(
                        range(entries["entry_num"]), desc="Evaluating"
                    ):
                        # Log info
                        rospy.loginfo(
                            f"==== Evaluation {idx}/{entries['entry_num']} ===="
                        )
                        rospy.loginfo(
                            "Evaluating with caod_head=%s, seg_head=%s, feature_encode_head=%s",
                            caod_head,
                            seg_head,
                            feature_encode_head,
                        )
                        ###########################################################################################################
                        rgb_path = entries["color_files"][idx]
                        instance_path = entries["2d-instance-filt_files"][idx]
                        label_path = entries["2d-label-filt_files"][idx]
                        # Now we have:
                        # rgb_path <str> : Path to the RGB image
                        # instance_path <str> : Path to the 2D instance segmentation image <0-N(8bit)>
                        # label_path <str> : Path to the 2D label segmentation image <0-1357(16bit)>
                        # id_category <dict> : Mapping from label IDs to category names <{1:{"raw_category":"tools", "category":"tool"}, ...}>
                        ################################################# Forward #################################################
                        # Read image
                        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB
                        if rgb is None:
                            rospy.logerr("Failed to read image: %s", rgb_path)
                            continue
                        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
                        # Assemble request
                        req = SingleExperimentRequest()
                        req.rgb_msg = rgb_msg
                        req.caod_head = caod_head
                        req.seg_head = seg_head
                        req.feature_encode_head = feature_encode_head
                        req.masks_merge_iou_threshold = self.masks_merge_iou_threshold
                        req.masks_merge_contain_threshold = (
                            self.masks_merge_contain_threshold
                        )
                        # Call service
                        resp = self.forward_pipeline_client(req)
                        time_dict = {
                            "forward": resp.time_forward,
                            "copy_data": resp.time_copy_data,
                            "seg_full": resp.time_seg_full,
                            "seg_bbox": resp.time_seg_bbox,
                            "obj_det_caod": resp.time_obj_det_caod,
                            "gen_tag_ram": resp.time_gen_tag_ram,
                            "obj_det_yolo": resp.time_obj_det_yolo,
                            "merge_masks": resp.time_merge_masks,
                            "extract_bboxes": resp.time_extract_bboxes,
                            "encode_images_clip": resp.time_encode_images_clip,
                            "encode_images_dinov2": resp.time_encode_images_dinov2,
                            "encode_images_dinov3": resp.time_encode_images_dinov3,
                        }
                        if not resp.success:
                            rospy.logerr("Service failed: %s", resp.message)
                            continue
                        # Log info
                        rospy.loginfo("==== Forward Time (s) ====")
                        for k, v in time_dict.items():
                            rospy.loginfo(f"{k}: {v:.4f}")
                        ############################################################################################################
                        seg_masks_np = np.array(
                            [
                                self.cv_bridge.imgmsg_to_cv2(mask_msg, "mono8")
                                for mask_msg in resp.seg_masks
                            ]
                        )
                        semantic_fts_np = np.array(
                            resp.semantic_fts, dtype=np.float32
                        ).reshape(-1, resp.semantic_ft_dim)
                        # Now we have:
                        # seg_masks_np <np.ndarray> : Merged segmentation masks (N, H, W) 二值掩码
                        # semantic_fts_np <np.ndarray> : Semantic features, shape (N, feature_dim)
                        ################################################# Evaluation ###############################################
                        # Read ground truth
                        inst_png = cv2.imread(
                            instance_path, cv2.IMREAD_UNCHANGED
                        )  # uint8
                        label_png = cv2.imread(
                            label_path, cv2.IMREAD_UNCHANGED
                        )  # uint16
                        if inst_png is None or label_png is None:
                            rospy.logwarn(
                                "Skip invalid GT pngs: %s / %s",
                                instance_path,
                                label_path,
                            )
                            continue
                        # Build id2name
                        id2name = make_id2name(id_category)
                        # Text encoder
                        text_encoder = self._text_encoder_for_feature(
                            feature_encode_head
                        )
                        # Compute metrics
                        combo_key = (caod_head, seg_head, feature_encode_head)
                        if combo_key not in self.metrics_by_combo:
                            self.metrics_by_combo[combo_key] = MetricsAggregator()
                            self.forward_times_by_combo[combo_key] = []
                            self.ros_distri_times_by_combo[combo_key] = []
                        metrics, vis_retr_imgs = self.smm.compute(
                            pred_masks=seg_masks_np,  # np.ndarray (N, H, W) 二值掩码
                            pred_feats=semantic_fts_np,  # np.ndarray (N, feature_dim)
                            instance_png=inst_png,  # 只用于 seg 指标
                            label_png=label_png,  # 只用于 retr 指标
                            text_encoder=text_encoder,
                            id2name=id2name,
                            color_image=rgb if self.show_vis_img else None,
                        )
                        if metrics is None:
                            rospy.logwarn(
                                "Metrics is None, likely due to no GT instances; skip."
                            )
                            continue
                        self.metrics_by_combo[combo_key].update(metrics)
                        self.forward_times_by_combo[combo_key].append(
                            float(resp.time_forward)
                        )
                        del time_dict["forward"]
                        self.ros_distri_times_by_combo[combo_key].append(
                            max(time_dict.values())
                        )
                        # Log info
                        rospy.loginfo("==== Metrics ====")
                        for k, v in metrics.items():
                            rospy.loginfo(f"{k}: {v:.4f}")
                        if vis_retr_imgs and self.show_vis_img:
                            for i, vis_img in enumerate(vis_retr_imgs):
                                cv2.imshow(f"Visualization {i+1}", vis_img)
                            key = cv2.waitKey(0)
                            if key == ord("q"):  # 按 'q' 键退出后续可视化
                                self.show_vis_img = False
                            cv2.destroyAllWindows()
                        ############################################################################################################
                        # Visualize
                        self._visualize(rgb, seg_masks_np, inst_png, label_png)

        # After all frames are done, print summary
        rows = []
        for combo_key, agg in self.metrics_by_combo.items():
            summary = agg.summary()  # 带 /mean /std
            # FPS
            tfwd = self.forward_times_by_combo.get(combo_key, [])
            fps = fps_from_times(tfwd)
            tfwd_ros = self.ros_distri_times_by_combo.get(combo_key, [])
            fps_ros = fps_from_times(tfwd_ros)
            summary["eff/FPS-ros-distri"] = fps_ros
            summary["eff/FPS"] = fps

            rospy.loginfo("\n=== COMBO %s ===", str(combo_key))
            for k, v in summary.items():
                rospy.loginfo(f"{k}: {v:.4f}")

            row = {"caod": combo_key[0], "seg": combo_key[1], "feat": combo_key[2]}
            row.update(summary)
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            out_csv = os.path.join(
                self.scene_dir, f"{self.scannet_scene}_eval_summary.csv"
            )
            df.to_csv(out_csv, index=False)
            rospy.loginfo("Saved summary CSV to: %s", out_csv)


if __name__ == "__main__":
    try:
        node = Evaluation()
        node.run_eval()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
