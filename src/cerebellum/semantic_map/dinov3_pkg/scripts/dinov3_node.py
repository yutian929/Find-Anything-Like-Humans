import rospy
import torch
from PIL import Image
from cv_bridge import CvBridge
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.data.transforms import make_classification_eval_transform
from dinov3_pkg.srv import DINOv3, DINOv3Response
import numpy as np


class DINOv3Node:
    def __init__(self):
        rospy.init_node("dinov3_node")

        # Load DINOv3 model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
        self.model = self.model.to(self.device)
        # self.model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(self.device)
        # self.tokenizer = get_tokenizer()
        self.image_preprocess = make_classification_eval_transform()
        self.bridge = CvBridge()

        rospy.loginfo("DINOv3 model loaded successfully.")

        # Initialize ROS service
        self.dinov3_service = rospy.Service(
            "dinov3", DINOv3, self.handle_dinov3_request
        )
        rospy.loginfo("DINOv3 ROS service initialized.")

    def _encode_text(self, text):
        """Encode a single text string."""
        tokenized_text = self.tokenizer.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text)
        return text_features.cpu().numpy()

    def _encode_image(self, ros_image):
        """Encode a single image from a ROS Image message."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            pil_image = Image.fromarray(cv_image)
            image_tensor = self.image_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
            return image_features.cpu().numpy()
        except Exception as e:
            rospy.logerr(f"Failed to encode image: {str(e)}")
            return None

    def _encode_images(self, ros_images):
        """Batch encode multiple images from ROS Image messages."""
        image_tensors = []
        for ros_image in ros_images:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
                pil_image = Image.fromarray(cv_image)
                image_tensor = self.image_preprocess(pil_image)
                image_tensors.append(image_tensor)
            except Exception as e:
                rospy.logwarn(f"Skipping invalid image: {str(e)}")
        if not image_tensors:
            return None
        batch_tensor = torch.stack(image_tensors).to(self.device)
        with torch.no_grad():
            batch_features = self.model.encode_image(batch_tensor)
        return batch_features.cpu().numpy()

    def handle_dinov3_request(self, req):
        """Handle incoming DINOv3 service requests."""
        mode = req.mode
        if mode == "encode_text":
            text = req.text
            text_features = self._encode_text(text)
            if text_features is not None:
                return DINOv3Response(
                    dinov3_fts=text_features.flatten().tolist(),
                    dinov3_ft_dim=text_features.shape[1],
                )
            else:
                rospy.logerr("Text encoding failed.")
                return DINOv3Response(dinov3_fts=[], dinov3_ft_dim=0)

        elif mode == "encode_image":
            image = req.image
            image_features = self._encode_image(image)
            if image_features is not None:
                return DINOv3Response(
                    dinov3_fts=image_features.flatten().tolist(),
                    dinov3_ft_dim=image_features.shape[1],
                )
            else:
                rospy.logerr("Image encoding failed.")
                return DINOv3Response(dinov3_fts=[], dinov3_ft_dim=0)

        elif mode == "encode_images":
            images = req.images
            batch_features = self._encode_images(images)
            if batch_features is not None:
                return DINOv3Response(
                    dinov3_fts=batch_features.flatten().tolist(),
                    dinov3_ft_dim=batch_features.shape[1],
                )
            else:
                rospy.logerr("Batch image encoding failed.")
                return DINOv3Response(dinov3_fts=[], dinov3_ft_dim=0)

        else:
            rospy.logerr(f"Invalid mode: {mode}")
            return DINOv3Response(dinov3_fts=[], dinov3_ft_dim=0)


if __name__ == "__main__":
    try:
        node = DINOv3Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
