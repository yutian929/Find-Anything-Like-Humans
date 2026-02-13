import cv2
import numpy as np
import os


def merge_images_with_labels(image_paths, output_path="merged_result.jpg"):
    """
    读取多个图片，在每张图片上标注名称，然后水平合并成一张图片

    参数:
        image_paths: 图片路径列表
        output_path: 输出的合并图片路径
    """
    images = []
    max_height = 0
    total_width = 0

    # 读取所有图片并添加标签
    for img_path in image_paths:
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 获取文件名（不含路径）
        img_name = os.path.basename(img_path)

        # 在图片底部添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        font_thickness = 3
        text_color = (0, 255, 0)  # BGR格式

        # 计算文本大小以便正确定位
        text_size, _ = cv2.getTextSize(img_name, font, font_scale, font_thickness)
        # text_x/y, 位于图片左上角
        text_x = 50
        text_y = 50

        # 添加文本到图片
        cv2.putText(
            img,
            img_name,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

        # 保存处理后的图片
        images.append(img)

        # 更新最大高度和总宽度
        max_height = max(max_height, img.shape[0])
        total_width += img.shape[1]

    if not images:
        print("没有有效图片可合并")
        return False

    # 创建合并图片的画布
    merged_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # 将图片拼接到画布上
    current_x = 0
    for img in images:
        h, w, _ = img.shape
        merged_img[0:h, current_x : current_x + w] = img
        current_x += w

    # 保存合并后的图片
    cv2.imwrite(output_path, merged_img)
    print(f"合并图片已保存至: {output_path}")
    return True


if __name__ == "__main__":
    image_paths = [
        "refrigerator_xp.jpg",
        "refrigerator_yn.jpg",
        "refrigerator_yp.jpg",
    ]

    merge_images_with_labels(image_paths, output_path="merged_refrigerator.jpg")

    image_paths = [
        "oven_yn.jpg",
        "oven_yp.jpg",
        "oven_xn.jpg",
    ]

    merge_images_with_labels(image_paths, output_path="merged_oven.jpg")
