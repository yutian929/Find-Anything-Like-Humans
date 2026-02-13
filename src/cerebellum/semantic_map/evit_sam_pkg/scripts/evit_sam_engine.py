import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


class EvitSAMEngine:
    def __init__(
        self,
        model_type: str = "efficientvit-sam-l1",
        ckpt_path: str = "weights/efficientvit_sam_l1.pt",
        device="cpu",
    ):
        # 创建模型并载入 checkpoint
        self.device = device
        self.model = (
            create_efficientvit_sam_model(model_type, True, ckpt_path).to(device).eval()
        )
        # 初始化 predictor
        self.predictor = EfficientViTSamPredictor(self.model)

    def set_image(self, image: np.ndarray):
        """
        输入一张图像，HWC格式，RGB。设置后可用于多次box预测。
        """
        assert image.ndim == 3 and image.shape[2] == 3, "image must be HWC format (RGB)"
        self.predictor.set_image(image)

    def segment_everything(self, image: np.ndarray):
        raise NotImplementedError(
            "EvitSAMEngine does not support full image segmentation. Use FastSAMEngine for that."
        )

    def segment_by_points(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray
    ):
        raise NotImplementedError(
            "EvitSAMEngine does not support point-based segmentation. Use FastSAMEngine for that."
        )

    def segment_by_boxes(self, image: np.ndarray, boxes: np.ndarray):
        """
        批量输入 bounding boxes (N, 4)，xyxy 格式。
        返回: (N, H, W) 的 masks。
        """
        assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be Nx4 shape"
        self.set_image(image)

        # 转换到模型输入格式
        transformed_boxes = self.predictor.transform.apply_boxes(
            boxes, self.predictor.original_size
        )
        boxes_torch = torch.as_tensor(
            transformed_boxes, dtype=torch.float, device=self.device
        )

        # 进行预测
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes_torch,
            multimask_output=False,
        )

        # 输出为 (N, H, W) 的 numpy bool 数组
        return masks.squeeze(1).cpu().numpy().astype(bool)


if __name__ == "__main__":
    import cv2

    # 加载图像（RGB）
    img = cv2.imread("328.png")
    img = cv2.resize(img, (640, 480))

    # 定义 box，例如：[x1, y1, x2, y2]
    boxes = np.array(
        [
            [100, 120, 200, 250],
            [300, 100, 400, 200],
        ]
    )

    engine = EvitSAMEngine()
    masks = engine.segment_by_boxes(img, boxes)

    print("输出 masks shape:", masks.shape)  # (N, H, W)
    # 可视化结果
    for i, mask in enumerate(masks):
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imshow(f"Mask {i}", mask_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
