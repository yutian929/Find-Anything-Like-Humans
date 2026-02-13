import torch
from alpha_clip_ import alpha_clip
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image
from torchvision import transforms
from sensor_msgs.msg import Image as RosImage

# Import the service
from alpha_clip_pkg.srv import AlphaCLIP, AlphaCLIPResponse


class AlphaClipNode:
    def __init__(self):
        rospy.init_node("alpha_clip_node")
        self.bridge = CvBridge()

        # Load model parameters
        model_type = rospy.get_param("~model_type", "ViT-B/16")
        checkpoint_path = rospy.get_param(
            "~checkpoint_path", "./weights/clip_b16_grit1m_fultune_8xe.pth"
        )

        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"AlphaCLIP using device: {self.device}")

        # Load model
        rospy.loginfo(
            f"Loading AlphaCLIP model {model_type} from checkpoint: {checkpoint_path}"
        )
        try:
            self.model, self.preprocess = alpha_clip.load(
                model_type,
                alpha_vision_ckpt_pth=checkpoint_path,
                device=self.device,
            )
            rospy.loginfo("AlphaCLIP model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load AlphaCLIP model: {e}")
            raise

        # Setup mask transform
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224)
                ),  # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26),
            ]
        )

        # Create service
        self.service = rospy.Service(
            "alpha_clip", AlphaCLIP, self.handle_alpha_clip_request
        )
        rospy.loginfo("AlphaCLIP service is ready")

    def handle_alpha_clip_request(self, req):
        """Handle AlphaCLIP service requests"""
        try:
            if req.mode == "encode_text":
                # Encode text
                # rospy.loginfo(f"Encoding text: {req.text}")
                features = self._encode_text(req.text)

            elif req.mode == "encode_alpha_image":
                # Encode image with masks
                # rospy.loginfo("Encoding image with alpha masks")
                features = self._encode_image_with_masks(req.rgb_img, req.alpha_masks)

            else:
                rospy.logerr(f"Unknown mode: {req.mode}")
                return None

            # Create response with features
            response = AlphaCLIPResponse()
            response.aclip_fts = features.flatten().astype(float).tolist()
            response.aclip_ft_dim = (
                features.shape[1] if len(features.shape) > 1 else len(features)
            )
            return response

        except Exception as e:
            rospy.logerr(f"Error handling AlphaCLIP request: {e}")
            return None

    def _encode_text(self, text):
        """Encode text into features"""
        with torch.no_grad():
            text_tokens = alpha_clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def _encode_image_with_masks(self, rgb_img_msg, alpha_masks_array):
        """Encode image with multiple mask images into features"""
        # Convert ROS Image to PIL Image
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_img_msg, "rgb8")
        pil_img = Image.fromarray(cv_rgb)

        # Process image once (can be reused for all masks)
        processed_image = self.preprocess(pil_img).unsqueeze(0).half().to(self.device)

        # Process each mask and collect features
        all_features = []

        # Check if we have any masks
        if not alpha_masks_array:
            rospy.logwarn("No masks provided")
            return np.empty((0, 512))  # Assuming feature dim is 512

        num_masks = len(alpha_masks_array)
        # rospy.loginfo(f"Processing {num_masks} masks")

        for i, mask_msg in enumerate(alpha_masks_array):
            try:
                # Convert mask to binary
                cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
                binary_mask = cv_mask > 0  # Convert grayscale to binary

                # Transform mask
                alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
                alpha = alpha.half().to(self.device).unsqueeze(dim=0)

                # Get image features for this mask
                with torch.no_grad():
                    image_features = self.model.visual(processed_image, alpha)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    all_features.append(image_features.cpu().numpy())

            except Exception as e:
                rospy.logerr(f"Error processing mask {i}: {e}")
                continue

        # Stack all features
        if all_features:
            return np.vstack(all_features)
        else:
            rospy.logwarn("No valid masks processed, returning empty features")
            return np.empty((0, 512))  # Assuming feature dim is 512


if __name__ == "__main__":
    try:
        node = AlphaClipNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error starting AlphaCLIP node: {e}")
