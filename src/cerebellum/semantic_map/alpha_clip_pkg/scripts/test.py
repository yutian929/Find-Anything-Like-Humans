import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage
import os

# Import service
from alpha_clip_pkg.srv import AlphaCLIP


class AlphaClipTester:
    def __init__(self):
        rospy.init_node("alpha_clip_tester")
        self.bridge = CvBridge()

        # Wait for AlphaCLIP service
        rospy.loginfo("Waiting for AlphaCLIP service...")
        rospy.wait_for_service("alpha_clip")
        self.alpha_clip_client = rospy.ServiceProxy("alpha_clip", AlphaCLIP)
        rospy.loginfo("Connected to AlphaCLIP service")

        # Test paths - update these to your actual test image paths
        self.test_img_path = rospy.get_param(
            "~test_img_path", "./alpha_clip_/examples/image.png"
        )
        self.test_mask_path = rospy.get_param(
            "~test_mask_path", "./alpha_clip_/examples/dress_mask.png"
        )

        # Example text prompts
        self.test_prompts = [
            "a gorgeously dressed woman",
            "a purple sleeveless dress",
            "bouquet of pink flowers",
        ]

    def load_and_convert_images(self):
        """Load test images and convert to ROS messages"""
        rospy.loginfo(f"Loading test image from {self.test_img_path}")
        rospy.loginfo(f"Loading test mask from {self.test_mask_path}")

        if not os.path.exists(self.test_img_path) or not os.path.exists(
            self.test_mask_path
        ):
            rospy.logerr(
                f"Test images not found! Please check paths: {self.test_img_path}, {self.test_mask_path}"
            )
            return None, None

        # Load image
        pil_image = PILImage.open(self.test_img_path).convert("RGB")
        cv_image = np.array(pil_image)

        # Load mask
        mask = np.array(PILImage.open(self.test_mask_path))

        # Convert mask to binary
        if len(mask.shape) == 2:
            binary_mask = mask == 255
        elif len(mask.shape) == 3:
            binary_mask = mask[:, :, 0] == 255
        else:
            rospy.logerr(f"Unexpected mask shape: {mask.shape}")
            return None, None

        # Convert binary mask to 8-bit image for ROS
        cv_mask = (binary_mask * 255).astype(np.uint8)

        # Convert to ROS messages
        rgb_msg = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
        mask_msg = self.bridge.cv2_to_imgmsg(cv_mask, "mono8")

        return rgb_msg, mask_msg

    def test_service(self):
        """Test the AlphaCLIP service with images and text"""
        try:
            # Load test images
            rgb_msg, mask_msg = self.load_and_convert_images()
            if rgb_msg is None or mask_msg is None:
                return

            # 1. Encode masked image
            rospy.loginfo("Encoding masked image...")
            image_response = self.alpha_clip_client(
                mode="encode_alpha_image",
                rgb_img=rgb_msg,
                alpha_masks=[mask_msg],  # List of masks
                # text=""
            )

            image_features = np.array(image_response.aclip_fts).reshape(
                -1, image_response.aclip_ft_dim
            )
            rospy.loginfo(f"Image features shape: {image_features.shape}")

            # 2. Encode text prompts
            text_features_list = []
            for prompt in self.test_prompts:
                rospy.loginfo(f"Encoding text: '{prompt}'")
                text_response = self.alpha_clip_client(
                    mode="encode_text",
                    # rgb_img=rgb_msg,  # Not used for text encoding
                    # alpha_masks=mask_msg,  # Not used for text encoding
                    text=prompt,
                )
                text_features = np.array(text_response.aclip_fts).reshape(
                    text_response.aclip_ft_dim,
                )
                text_features_list.append(text_features)

            # Stack all text features
            all_text_features = np.vstack(text_features_list)

            # 3. Compute similarities
            similarities = 100.0 * np.dot(image_features, all_text_features.T)
            softmax_similarities = self._softmax(similarities[0])

            # 4. Print results
            rospy.loginfo("=== AlphaCLIP Test Results ===")
            rospy.loginfo(f"Raw similarities: {similarities[0]}")
            rospy.loginfo(f"Softmax similarities: {softmax_similarities}")

            for i, prompt in enumerate(self.test_prompts):
                rospy.loginfo(f"  '{prompt}': {softmax_similarities[i]*100:.2f}%")

            rospy.loginfo("Test completed successfully!")

        except Exception as e:
            rospy.logerr(f"Test failed with error: {e}")

    def _softmax(self, x):
        """Compute softmax values for array x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == "__main__":
    try:
        tester = AlphaClipTester()
        tester.test_service()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in test node: {e}")
