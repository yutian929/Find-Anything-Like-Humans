import cv2
import numpy as np
import random
import os
from typing import List


def load_image(image_path: str) -> np.ndarray:
    """读取RGB图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式


def load_instance_mask(mask_path: str) -> np.ndarray:
    """加载实例分割掩码"""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码：{mask_path}")
    return mask  # 8-bit 实例ID图像


def load_label_mask(mask_path: str) -> np.ndarray:
    """加载语义分割掩码"""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取语义掩码：{mask_path}")
    return mask  # 16-bit 类别ID图像


def draw_id_on_mask(
    image: np.ndarray, mask: np.ndarray, instance_id: int, color: List[int]
):
    """
    在给定掩码上绘制对应的实例 ID 或 类别 ID。
    - 输入：image 图像，mask 掩码，instance_id 要显示的 ID，color 掩码的颜色
    - 输出：绘制了 ID 的图像
    """
    # 计算掩码的边界框，获得其中心点
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # 获取最大的轮廓（应该是当前实例掩码的轮廓）
        contour = max(contours, key=cv2.contourArea)
        # 计算该轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 计算中心点
        center_x, center_y = x + w // 2, y + h // 2
        # 在该位置绘制文本
        cv2.putText(
            image,
            str(instance_id),
            (center_x - 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def visualize_instance_and_label_segmentation(
    image_path: str, instance_mask_path: str, label_mask_path: str
):
    """
    可视化实例分割和语义分割结果：
    - 输入：图像路径、实例掩码路径、语义掩码路径
    - 输出：图像与实例掩码和语义掩码叠加的结果，并在每个掩码上方绘制 ID
    """
    # 加载图像、实例掩码和语义掩码
    image = load_image(image_path)
    instance_mask = load_instance_mask(instance_mask_path)
    label_mask = load_label_mask(label_mask_path)

    # 创建一个颜色映射，确保每个实例和类别都有唯一的颜色
    unique_instances = np.unique(instance_mask)
    instance_color_map = {}
    for instance_id in unique_instances:
        if instance_id == 0:
            continue  # 背景不需要颜色
        # 随机生成颜色
        instance_color_map[instance_id] = [random.randint(0, 255) for _ in range(3)]

    # 创建另一个颜色映射，用于语义分割
    unique_labels = np.unique(label_mask)
    label_color_map = {}
    for label_id in unique_labels:
        if label_id == 0:
            continue  # 背景不需要颜色
        # 随机生成颜色
        label_color_map[label_id] = [random.randint(0, 255) for _ in range(3)]

    # 创建一个与图像大小相同的空白图像，用于叠加掩码
    overlay_image_instance = image.copy()
    overlay_image_label = image.copy()

    # 迭代每个实例，将其掩码叠加到图像上并显示实例 ID
    for instance_id, color in instance_color_map.items():
        # 获取该实例的掩码位置
        instance_mask_pixels = instance_mask == instance_id
        overlay_image_instance[instance_mask_pixels] = color  # 用随机颜色填充该实例的区域
        draw_id_on_mask(
            overlay_image_instance, instance_mask_pixels, instance_id, [255, 255, 255]
        )  # 在该实例上绘制 ID

    # 迭代每个语义标签，将其掩码叠加到图像上并显示类别 ID
    for label_id, color in label_color_map.items():
        # 获取该语义标签的掩码位置
        label_mask_pixels = label_mask == label_id
        overlay_image_label[label_mask_pixels] = color  # 用随机颜色填充该标签的区域
        draw_id_on_mask(
            overlay_image_label, label_mask_pixels, label_id, [255, 255, 255]
        )  # 在该标签上绘制 ID

    # 显示图像和叠加的掩码
    cv2.imshow("Original Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow(
        "Instance Segmentation", cv2.cvtColor(overlay_image_instance, cv2.COLOR_RGB2BGR)
    )
    cv2.imshow(
        "Label Segmentation", cv2.cvtColor(overlay_image_label, cv2.COLOR_RGB2BGR)
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 输入图像路径和实例、语义分割掩码路径
    for idx in range(10):
        suffix = idx * 20
        image_path = (
            f"/home/yutian/下载/scannetv2/scene0200_00/color/{suffix}.jpg"  # 替换为实际图像路径
        )
        instance_mask_path = f"/home/yutian/下载/scannetv2/scene0200_00/scene0200_00_2d-instance-filt/instance-filt/{suffix}.png"  # 替换为实例分割掩码路径
        label_mask_path = f"/home/yutian/下载/scannetv2/scene0200_00/scene0200_00_2d-label-filt/label-filt/{suffix}.png"  # 替换为语义分割掩码路径

        visualize_instance_and_label_segmentation(
            image_path, instance_mask_path, label_mask_path
        )
