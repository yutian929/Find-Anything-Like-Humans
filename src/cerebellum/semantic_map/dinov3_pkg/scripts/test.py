import rospy
from dinov3_pkg.srv import DINOv3, DINOv3Request
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import os


def create_ros_image_from_file(image_path):
    """Convert a local image file to a ROS Image message."""
    bridge = CvBridge()
    pil_image = PILImage.open(image_path).convert("RGB")
    cv_image = np.array(pil_image)
    ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
    return ros_image


def calculate_similarity(text_feature, image_feature):
    """Calculate similarity between text and image features."""
    text_feature = np.array(text_feature)
    image_feature = np.array(image_feature)
    similarity = np.dot(image_feature, text_feature) / (
        np.linalg.norm(image_feature) * np.linalg.norm(text_feature)
    )
    return similarity


def test_dinov3_service():
    rospy.init_node("test_dinov3_node", anonymous=True)
    rospy.wait_for_service("dinov3")
    dinov3_service = rospy.ServiceProxy("dinov3", DINOv3)
    cat_img_path = rospy.get_param("~cat_img_path", "/path/to/your/cat_image.jpg")
    dog_img_path = rospy.get_param("~dog_img_path", "/path/to/your/dog_image.jpg")

    # Test encode_text
    try:
        text_request = DINOv3Request()
        text_request.mode = "encode_text"
        text_request.text = "A photo of a dog"
        text_response = dinov3_service(text_request)
        text_feature = text_response.dinov3_fts
        rospy.loginfo(
            f"Text encoding result: {text_feature[:10]}... (dim: {text_response.dinov3_ft_dim})"
        )
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call dinov3 service for encode_text: {e}")
        return

    # Test encode_image
    try:
        if not os.path.exists(cat_img_path):
            rospy.logerr(f"Image file not found: {cat_img_path}")
            return
        ros_image = create_ros_image_from_file(cat_img_path)

        image_request = DINOv3Request()
        image_request.mode = "encode_image"
        image_request.image = ros_image
        image_response = dinov3_service(image_request)
        cat_image_feature = image_response.dinov3_fts
        rospy.loginfo(
            f"Cat image encoding result: {cat_image_feature[:10]}... (dim: {image_response.dinov3_ft_dim})"
        )
        # Calculate similarity between text and cat image
        similarity = calculate_similarity(text_feature, cat_image_feature)
        rospy.loginfo(f"Similarity between text and cat image: {similarity:.4f}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call dinov3 service for encode_image: {e}")

    # Test encode_images
    try:
        image_paths = [cat_img_path, dog_img_path]
        ros_images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                rospy.logwarn(f"Image file not found: {image_path}")
                continue
            ros_images.append(create_ros_image_from_file(image_path))

        if not ros_images:
            rospy.logerr("No valid images found for encode_images test.")
            return

        images_request = DINOv3Request()
        images_request.mode = "encode_images"
        images_request.images = ros_images
        images_response = dinov3_service(images_request)
        image_features = np.array(images_response.dinov3_fts).reshape(
            -1, images_response.dinov3_ft_dim
        )
        rospy.loginfo(
            f"Images encoding result: {image_features[:10]}... (dim: {images_response.dinov3_ft_dim})"
        )

        # Calculate similarity between text and each image
        for i, image_feature in enumerate(image_features):
            similarity = calculate_similarity(text_feature, image_feature)
            rospy.loginfo(f"Similarity with image {image_paths[i]}: {similarity:.4f}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call dinov3 service for encode_images: {e}")


if __name__ == "__main__":
    try:
        test_dinov3_service()
    except rospy.ROSInterruptException:
        pass
