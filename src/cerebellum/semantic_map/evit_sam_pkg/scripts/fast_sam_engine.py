from ultralytics import FastSAM
import numpy as np
import cv2


class FastSAMEngine:
    def __init__(
        self,
        model_path="weights/FastSAM-x.pt",
        device="cuda",
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    ):
        self.model = FastSAM(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _resize_masks(self, masks, target_shape):
        """
        Resize all masks to the target shape (height, width)
        """
        resized_masks = []
        h, w = target_shape
        for mask in masks:
            mask_np = mask.astype(np.uint8) * 255
            resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized > 0)
        return np.stack(resized_masks)

    def segment_everything(self, image):
        results = self.model(
            image,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        if not results[0].masks:
            return None
        masks = results[0].masks.data.cpu().numpy()
        return self._resize_masks(masks, image.shape[:2])

    def segment_by_points(self, image, points, labels):
        results = self.model(
            image, device=self.device, points=points, labels=labels, imgsz=self.imgsz
        )
        masks = results[0].masks.data.cpu().numpy()
        return self._resize_masks(masks, image.shape[:2])

    def segment_by_boxes(self, image, bboxes):
        results = self.model(image, device=self.device, bboxes=bboxes, imgsz=self.imgsz)
        masks = results[0].masks.data.cpu().numpy()
        return self._resize_masks(masks, image.shape[:2])


if __name__ == "__main__":
    import time

    # 测试代码
    engine = FastSAMEngine()

    # 读取测试图像
    image = cv2.imread("dog.jpg")
    image_rgb = cv2.resize(image, (640, 480))  # 调整图像大小
    cv2.imshow("Test Image", image_rgb)
    cv2.waitKey(0)

    # 全图像分割
    start = time.time()
    for _ in range(10):
        masks = engine.segment_everything(image_rgb)
    elapsed_time = time.time() - start
    print("全图像分割结果:", masks.shape)
    print("平均分割耗时:", elapsed_time / 10)

    # # 点提示分割
    # points = [[100, 100], [200, 200]]
    # labels = [1, 0]
    # masks_points = engine.segment_by_points(image_rgb, points, labels)
    # print("点提示分割结果:", masks_points.shape)

    # # 框提示分割
    # bboxes = [[100, 100, 200, 250], [300, 150, 400, 300]]
    # masks_boxes = engine.segment_by_boxes(image_rgb, bboxes)
    # print("框提示分割结果:", masks_boxes.shape)
