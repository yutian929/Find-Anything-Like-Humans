# scene_memory_metrics.py
# -*- coding: utf-8 -*-
"""
SceneMemoryMetrics: 单帧评测器（numpy 版本）
==================================================
- 输入的预测掩码 pred_masks 采用 numpy 数组: (N, H, W)，元素非零即视为正；
  不需要也不会在本模块中转换为 list。

- 实例覆盖质量（只用 instance PNG）：
    seg/recall                —— 贪心 IoU≥阈值 的一对一匹配召回率
    seg/mean_iou              —— 所有匹配对 IoU 的平均
    seg/mask_count_ratio      —— 预测掩码数 / GT 实例数（>1 可能碎片化，<1 可能漏检）

- 语义检索能力（只用 label PNG，类中心检索）：
    retr/top1, retr/top3      —— 每个出现的 GT 类别作为一次查询；前K命中率
    retr/mAP                  —— 对每个类别的二分类 AP 求平均

- 效率：
    fps_from_times            —— 由端到端耗时（秒）计算平均 FPS

- 结果聚合：
    MetricsAggregator         —— 累积每帧指标，输出 mean/std

使用方法（与 Evaluation 节点对接）：
--------------------------------------------------
from scene_memory_metrics import (
    SceneMemoryMetrics, MetricsAggregator, fps_from_times, make_id2name
)

smm = SceneMemoryMetrics(seg_iou_thr=0.5, retr_iou_thr=0.5,
                         retrieval_ks=(1,3),
                         prompt_templates=("a photo of a {}", "a {} in a room"))

metrics = smm.compute(
    pred_masks=seg_masks_np,        # np.ndarray, shape=(N, H, W), uint8/bool
    pred_feats=semantic_fts_np,     # np.ndarray, shape=(N, D), float32
    instance_png=inst_png,          # np.ndarray, shape=(H, W), uint8 (0=bg, >0=instance id)
    label_png=label_png,            # np.ndarray, shape=(H, W), uint16 (0=bg, >0=class id)
    text_encoder=text_encoder,      # callable(str)->np.ndarray(D,)
    id2name=make_id2name(id_category_dict),
)

—— 返回一个 dict，包括上述各项指标。
"""

from typing import Dict, Callable, Tuple, Iterable, List
import cv2
import numpy as np
from collections import defaultdict


# ==================================================
# 工具函数（ID->名字；AP；归一化；等）
# ==================================================


def make_id2name(id_category_dict: Dict) -> Callable[[int], str]:
    """
    将 {id: {'category', 'raw_category'}} 映射，封装为稳健的 id->name 函数。
    - 优先返回 'category'，没有则回退 'raw_category'；再没有返回 'object'。
    - 兼容键是 str 或 int 的情况。
    """

    def _id2name(cid: int) -> str:
        key_s, key_i = str(cid), cid
        if key_s in id_category_dict:
            d = id_category_dict[key_s]
            return d.get("category", d.get("raw_category", "object"))
        if key_i in id_category_dict:
            d = id_category_dict[key_i]
            return d.get("category", d.get("raw_category", "object"))
        return "object"

    return _id2name


def l2_normalize(x: np.ndarray, axis=-1, eps=1e-8) -> np.ndarray:
    """L2 归一化（避免除零）。"""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    二分类 AP（逐点累积法，等价于 PR 曲线面积）。
    - y_true: [N] ∈ {0,1}
    - y_score: [N] 分数，越大越正
    """
    order = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order]
    tp = 0
    fp = 0
    P = int(np.sum(y_true))
    if P == 0:
        return 0.0
    precisions = []
    recalls = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / P)
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(0.0, r - prev_r)
        prev_r = r
    return float(ap)


# ==================================================
# 掩码相关（实例提取、类并集、IoU 相关）
# ==================================================


def _to_bool(mask: np.ndarray) -> np.ndarray:
    """将掩码转为 0/1。输入可为 uint8/uint16/bool，非零即 1。"""
    return (mask > 0).astype(np.uint8)


def extract_instance_stack(inst_png: np.ndarray) -> np.ndarray:
    """
    从 instance PNG（uint8, 0=背景, >0=实例ID）提取实例栈：
      返回 bool/uint8 栈：shape = (G, H, W)，每层为一个实例二值掩码（0/1）。
    说明：相比返回 list，返回栈便于后续用 numpy 向量化计算。
    """
    H, W = inst_png.shape[:2]
    ids = [int(i) for i in np.unique(inst_png) if i != 0]
    if len(ids) == 0:
        return np.zeros((0, H, W), dtype=np.uint8)
    stack = np.zeros((len(ids), H, W), dtype=np.uint8)
    for idx, iid in enumerate(ids):
        stack[idx] = (inst_png == iid).astype(np.uint8)
    return stack


def class_union_masks_from_label(label_png: np.ndarray) -> Dict[int, np.ndarray]:
    """
    从 label PNG（uint16, 0=背景, >0=类ID）构造“类并集掩码”：
    返回 dict: {class_id -> (H, W) uint8 0/1 掩码}
    """
    out = {}
    classes = [int(c) for c in np.unique(label_png) if c != 0]
    for cid in classes:
        out[cid] = (label_png == cid).astype(np.uint8)
    return out


def iou_matrix_np(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """
    计算预测与 GT 的 IoU 矩阵（向量化版本）。
    - pred_masks: (N, H, W)  0/1 或 0/255（非零视为1）
    - gt_masks  : (G, H, W)  0/1
    返回：
      M: (N, G) float32 的 IoU 矩阵
    原理：
      把 (N, H*W) 与 (G, H*W) 展平成二值矩阵，
      交集 = 按位与计数 = (二值乘法后求和)；
      并集 = sum(pred) + sum(gt) - 交集
    """
    if pred_masks.ndim != 3:
        raise ValueError("pred_masks 必须是 (N, H, W)")
    if gt_masks.ndim != 3:
        raise ValueError("gt_masks 必须是 (G, H, W)")

    N, H, W = pred_masks.shape
    G = gt_masks.shape[0]

    P = _to_bool(pred_masks).reshape(N, -1).astype(np.uint8)  # (N, HW)
    Gt = _to_bool(gt_masks).reshape(G, -1).astype(np.uint8)  # (G, HW)

    # 交集：矩阵乘 (N,HW) x (HW,G) -> (N,G)
    inter = (P.astype(np.uint32) @ Gt.T.astype(np.uint32)).astype(np.float32)

    # 并集：sum(P) + sum(Gt) - inter
    sum_p = P.sum(axis=1, dtype=np.uint32).astype(np.float32).reshape(N, 1)  # (N,1)
    sum_g = Gt.sum(axis=1, dtype=np.uint32).astype(np.float32).reshape(1, G)  # (1,G)
    union = sum_p + sum_g - inter
    union = np.maximum(union, 1.0)  # 避免除零

    M = inter / union
    return M.astype(np.float32)


def greedy_match_by_iou(
    M: np.ndarray, thr: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    基于 IoU 的贪心一对一匹配（和检测评测里的常用做法一致）。
    输入：
      M: (N, G) 的 IoU 矩阵（预测 N × GT G）
      thr: 匹配阈值，比如 0.5
    返回：
      matches: [(pi, gj, iou), ...]
      unmatched_pred: 未匹配预测索引列表
      unmatched_gt  : 未匹配 GT 索引列表
    """
    N, G = M.shape
    used_p, used_g = set(), set()
    matches = []

    while True:
        # 在尚未使用的行/列里找最大 IoU
        max_iou, pi, gj = -1.0, -1, -1
        for i in range(N):
            if i in used_p:
                continue
            # 对未使用的 GT 列找最大 IoU 的列索引
            # 这里为了代码直观性仍用循环，N/G 通常不大；如需极致效率可以做掩码向量化
            for j in range(G):
                if j in used_g:
                    continue
                val = float(M[i, j])
                if val > max_iou:
                    max_iou, pi, gj = val, i, j

        if max_iou < thr:
            break
        matches.append((pi, gj, max_iou))
        used_p.add(pi)
        used_g.add(gj)

    unmatched_pred = [i for i in range(N) if i not in used_p]
    unmatched_gt = [j for j in range(G) if j not in used_g]
    return matches, unmatched_pred, unmatched_gt


def iou_vs_single_mask(pred_masks: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    计算每个预测掩码与一个 GT 掩码的 IoU（向量化）。
    - pred_masks: (N, H, W) 0/1
    - gt_mask   : (H, W)   0/1
    返回：
      ious: (N,) float32
    """
    if pred_masks.ndim != 3:
        raise ValueError("pred_masks 必须是 (N, H, W)")
    if gt_mask.ndim != 2:
        raise ValueError("gt_mask 必须是 (H, W)")

    N, H, W = pred_masks.shape
    P = _to_bool(pred_masks).reshape(N, -1).astype(np.uint8)  # (N, HW)
    G = _to_bool(gt_mask).reshape(1, -1).astype(np.uint8)  # (1, HW)

    inter = (
        (P.astype(np.uint32) @ G.T.astype(np.uint32)).astype(np.float32).reshape(-1)
    )  # (N,)
    sum_p = P.sum(axis=1, dtype=np.uint32).astype(np.float32)  # (N,)
    sum_g = float(G.sum(dtype=np.uint32))
    union = sum_p + sum_g - inter
    union = np.maximum(union, 1.0)
    return (inter / union).astype(np.float32)


# ==================================================
# 指标计算（单帧）
# ==================================================


class SceneMemoryMetrics:
    """
    单帧评测器（numpy 版本）
    - 分割指标：只用 instance_png（实例 PNG）
    - 检索指标：只用 label_png（类并集，类中心检索）
    """

    def __init__(
        self,
        seg_iou_thr: float = 0.5,  # 分割匹配阈值
        retr_cov_thr: float = 0.5,  # 检索中“正样本覆盖率阈值”（类并集）
        retrieval_ks: Iterable[int] = (1, 3),
        prompt_prefix: str = "a photo of a ",
    ):
        self.seg_iou_thr = float(seg_iou_thr)
        self.retr_cov_thr = float(retr_cov_thr)
        self.retrieval_ks = tuple(int(k) for k in retrieval_ks)
        self.prompt_prefix = prompt_prefix

    def compute(
        self,
        pred_masks: np.ndarray,  # (N, H, W) uint8/bool —— 预测掩码（融合后）
        pred_feats: np.ndarray,  # (N, D) float32 —— 与 pred_masks 一一对应
        instance_png: np.ndarray,  # (H, W) uint8   —— GT 实例 PNG：0=背景, >0=实例ID
        label_png: np.ndarray,  # (H, W) uint16  —— GT 语义 PNG：0=背景, >0=类别ID
        text_encoder: Callable[[str], np.ndarray],  # 文本 → 向量 [D]
        id2name: Callable[[int], str],  # 类ID → 名称
        color_image: np.ndarray = None,  # (H, W, 3) 可选的彩色图像，用于可视化
    ) -> Dict[str, float]:
        """
        返回值：包含分割 + 检索 + 辅助统计的字典：
          分割：'seg/AP@{seg_iou_thr}'
          检索：'retr/top1', 'retr/top3', 'retr/mAP'
          统计：'count/pred', 'count/gt_inst', 'count/gt_cls'
        """
        # -------- 输入检查 & 对齐 --------
        if pred_masks.ndim != 3:
            raise ValueError("pred_masks 必须是 (N, H, W)")
        Np = pred_masks.shape[0]
        if Np > 0:
            if not (
                isinstance(pred_feats, np.ndarray)
                and pred_feats.ndim == 2
                and pred_feats.shape[0] == Np
            ):
                raise ValueError("pred_feats 必须为 (N, D) 并与 pred_masks 的 N 对齐")

        # -------- 1) 实例覆盖质量（只用 instance）--------
        gt_inst_stack = extract_instance_stack(instance_png)  # (G, H, W)
        if gt_inst_stack.shape[0] == 0:
            return {}, []  # 没有 GT 实例，无法评测
        # seg_metrics = self._segmentation_metrics_np(pred_masks, gt_inst_stack, self.seg_iou_thr)
        seg_metrics = self.average_precision_class_agnostic(
            pred_masks, gt_inst_stack, self.seg_iou_thr
        )

        # -------- 2) 语义检索能力（只用 label，类中心）--------
        retr_metrics, vis_retr_imgs = self._retrieval_class_centric_np(
            pred_masks=pred_masks,
            pred_feats=pred_feats,
            label_png=label_png,
            text_encoder=text_encoder,
            id2name=id2name,
            cov_thr=self.retr_cov_thr,
            ks=self.retrieval_ks,
            prompt_prefix=self.prompt_prefix,
            color_image=color_image,
        )

        out = {}
        out.update(seg_metrics)
        out.update(retr_metrics)
        # out["count/pred"] = float(Np)
        # out["count/gt_inst"] = float(gt_inst_stack.shape[0])
        # out["count/gt_cls"] = float(len([c for c in np.unique(label_png) if c != 0]))
        return out, vis_retr_imgs

    # ---------- 内部：分割（numpy 版本） ----------
    @staticmethod
    def average_precision_class_agnostic(
        pred_masks: np.ndarray,
        gt_inst_stack: np.ndarray,
        iou_thr: float = 0.5,
        scores: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        类无关实例分割的 AP 计算（COCO 风格，mask IoU）。
        - pred_masks: (N, H, W)
        - gt_inst_stack: (G, H, W)
        - iou_thr: IoU 阈值 (默认 0.5)
        - scores: (N,) 每个预测的置信度；如果 None 就全 1

        返回:
        {'AP@seg_iou_thr': IoU=seg_iou_thr时的AP}
        """
        N = pred_masks.shape[0]
        G = gt_inst_stack.shape[0]
        if N == 0:
            return {f"seg/AP@{iou_thr}": 0.0}

        if scores is None:
            scores = np.ones(N, dtype=np.float32)

        # IoU 矩阵 (N, G)
        ious = iou_matrix_np(pred_masks, gt_inst_stack)

        aps = []
        # 按分数排序
        order = np.argsort(-scores)
        matched_gt = set()
        tp, fp = [], []

        for idx in order:
            # 当前预测与所有未匹配 GT 的 IoU
            gt_idx = -1
            max_iou = iou_thr
            for j in range(G):
                if j in matched_gt:
                    continue
                if ious[idx, j] >= max_iou:
                    max_iou = ious[idx, j]
                    gt_idx = j
            if gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gt.add(gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / max(1, G)
        precision = tp / np.maximum(tp + fp, 1)

        # 逐点积分（插值法）
        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precision, recall):
            ap += p * (r - prev_r)
            prev_r = r

        aps.append(ap)

        return {f"seg/AP@{iou_thr}": float(np.mean(aps))}

    # @staticmethod
    # def _segmentation_metrics_np(
    #     pred_masks: np.ndarray, gt_inst_stack: np.ndarray, iou_thr: float
    # ) -> Dict[str, float]:
    #     """
    #     pred_masks: (N, H, W)
    #     gt_inst_stack: (G, H, W)
    #     """
    #     N = pred_masks.shape[0]
    #     G = gt_inst_stack.shape[0]
    #     if N == 0 and G == 0:
    #         return {"seg/recall": 0.0, "seg/mean_iou": 0.0, "seg/mask_count_ratio": 0.0}

    #     if N > 0 and G > 0:
    #         M = iou_matrix_np(pred_masks, gt_inst_stack)  # (N, G)
    #         matches, _, _ = greedy_match_by_iou(M, iou_thr)
    #     else:
    #         M = np.zeros((N, G), dtype=np.float32)
    #         matches = []

    #     tp = len(matches)
    #     recall = tp / max(1, G)
    #     mean_iou = float(np.mean([m[2] for m in matches])) if tp > 0 else 0.0
    #     mcr = N / max(1, G)

    #     return {
    #         "seg/recall": recall,
    #         "seg/mean_iou": mean_iou,
    #         "seg/mask_count_ratio": mcr,
    #     }

    # ---------- 内部：检索（类中心；numpy 版本） ----------
    @staticmethod
    def calc_coverage(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        计算 pred_mask 覆盖 gt_mask 的比例（交集/pred_mask面积）。
        - pred_mask: (H, W) 0/1
        - gt_mask  : (H, W) 0/1
        返回：
          coverage: float ∈ [0,1]
        """
        inter = float(np.sum((pred_mask > 0) & (gt_mask > 0), dtype=np.uint32))
        pred_area = float(np.sum(pred_mask > 0, dtype=np.uint32))
        if pred_area == 0:
            return 0.0
        return inter / pred_area

    def _retrieval_class_centric_np(
        self,
        pred_masks: np.ndarray,  # (N, H, W)
        pred_feats: np.ndarray,  # (N, D)
        label_png: np.ndarray,  # (H, W) uint16
        text_encoder: Callable[[str], np.ndarray],
        id2name: Callable[[int], str],
        cov_thr: float,  # 覆盖率阈值
        ks: Iterable[int],
        prompt_prefix: str,
        color_image: np.ndarray = None,  # (H, W, 3)
    ) -> Dict[str, float]:

        res = {
            "retr/top1_hits": 0,
            "retr/top3_hits": 0,
            "retr/top1_base": 0,
            "retr/top3_base": 0,
        }

        # 没有候选或没有类别，直接返回
        if pred_masks.shape[0] == 0:
            return res
        cdict = class_union_masks_from_label(label_png)  # {cid: (H, W)} uint8 0/1
        cids = sorted(cdict.keys())
        if len(cids) == 0:
            return res

        # 忽略
        ignore_cids = {1, 3, 41}
        cids = [cid for cid in cids if cid not in ignore_cids]

        # 1) 文本向量：L2 归一
        text_vecs = {}
        for cid in cids:
            name = id2name(cid)
            v = text_encoder(prompt_prefix + name).astype(np.float32)
            text_vecs[cid] = l2_normalize(v)

        # 2) 预测特征 L2 归一
        X = l2_normalize(pred_feats.astype(np.float32), axis=1)  # (N, D)

        # 3) 逐类计算 Top-K 命中
        hits = {k: 0 for k in ks}
        vis_retr_imgs = []

        for cid in cids:
            gt_mask = cdict[cid]  # (H, W)
            t = text_vecs[cid]  # (D,)

            # 相似度分数
            scores = (X @ t.reshape(-1, 1)).reshape(-1)  # (N,)
            order = np.argsort(-scores)  # 从大到小排序

            # --- 只取前 max(ks) 个 ---
            found_at = None
            for rank, idx in enumerate(order[: max(ks)], start=1):
                cov = SceneMemoryMetrics.calc_coverage(pred_masks[idx], gt_mask)
                if cov >= cov_thr:
                    found_at = rank
                    break

            # --- 更新 hits ---
            if found_at is not None:
                for k in ks:
                    if found_at <= k:  # 在 top-k 内找到了
                        hits[k] += 1

            # --- 可视化 (可选) ---
            if color_image is not None:
                H, W = 480, 640
                panels, labels = [], []

                # GT
                gt_overlay = color_image.copy()
                gt_overlay[gt_mask > 0] = (0, 255, 0)
                gt_overlay = cv2.resize(gt_overlay, (W, H))
                panels.append(gt_overlay)
                labels.append(f"GT {id2name(cid)}")

                # Top-K 预测 (统一红色)
                red = (0, 0, 255)
                for rank, idx in enumerate(order[: max(ks)], start=1):
                    pred_mask = pred_masks[idx]
                    topk_overlay = color_image.copy()
                    topk_overlay[pred_mask > 0] = red
                    topk_overlay = cv2.resize(topk_overlay, (W, H))
                    status = (
                        "YES" if found_at is not None and rank == found_at else "NO"
                    )
                    panels.append(topk_overlay)
                    labels.append(f"Pred {rank} {status}")

                # 只取前 4 张（GT + Top3）
                panels, labels = panels[:4], labels[:4]

                # --- 补黑图 ---
                while len(panels) < 4:
                    panels.append(np.zeros_like(panels[0]))
                    labels.append("")

                # --- 加文字 ---
                for img, text in zip(panels, labels):
                    if text:
                        cv2.putText(
                            img,
                            text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                # 拼接 2x2
                top_row = np.hstack(panels[:2])
                bottom_row = np.hstack(panels[2:])
                vis_retr_img = np.vstack([top_row, bottom_row])
                vis_retr_imgs.append(vis_retr_img)

        # 平均到类别
        num_classes = max(1, len(cids))
        for k in ks:
            res[f"retr/top{k}_hits"] = hits[k]
            res[f"retr/top{k}_base"] = num_classes

        return res, vis_retr_imgs


# ==================================================
# 指标聚合（跨帧/跨组合）
# ==================================================


class MetricsAggregator:
    """
    累积每帧的指标，最后导出 mean/std。
    用法：
      agg = MetricsAggregator()
      agg.update(frame_metrics)   # dict[str -> float]
      summary = agg.summary()     # dict[str -> float]，包含 xxx/mean 与 xxx/std
    """

    def __init__(self):
        self.buf = {
            "seg": {"seg/AP@0.5": []},
            "retr": {
                "retr/top1_hits": 0,
                "retr/top3_hits": 0,
                "retr/top1_base": 0,
                "retr/top3_base": 0,
            },
        }

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if "seg" in k:  # segmentation metrics, 'seg/AP@{seg_iou_thr}'
                self.buf["seg"][k].append(v)
            elif (
                "retr" in k
            ):  # retrieval metrics, 'retr/top1_hits/base', 'retr/top3_hits/base'
                if "hits" in k or "base" in k:
                    self.buf["retr"][k] += v

    def summary(self) -> Dict[str, float]:
        out = {}
        # 'seg/AP@0.5'
        list_seg_ap_50 = self.buf["seg"].get("seg/AP@0.5", [])
        out["seg/AP@0.5_mean"] = (
            float(np.mean(list_seg_ap_50)) if list_seg_ap_50 else 0.0
        )
        # 'retr/top1'
        out["retr/top1"] = self.buf["retr"]["retr/top1_hits"] / max(
            1, self.buf["retr"]["retr/top1_base"]
        )
        # 'retr/top3'
        out["retr/top3"] = self.buf["retr"]["retr/top3_hits"] / max(
            1, self.buf["retr"]["retr/top3_base"]
        )
        return out


# ==================================================
# 效率辅助
# ==================================================


def fps_from_times(seconds_list: List[float]) -> float:
    """
    根据多帧端到端耗时（秒）计算平均 FPS：
      FPS = 1 / mean(time_forward)
    """
    if not seconds_list:
        return 0.0
    mean_t = float(np.mean(seconds_list))
    return 1.0 / mean_t if mean_t > 1e-8 else 0.0
