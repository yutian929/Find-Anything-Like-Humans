import rospy
from clip_pkg.srv import CLIP, CLIPRequest
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import os


def create_ros_image_from_file(image_path):
    """将本地图片文件转换为 ROS Image 消息"""
    bridge = CvBridge()
    pil_image = PILImage.open(image_path).convert("RGB")
    cv_image = np.array(pil_image)
    ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
    return ros_image


def calculate_similarity(text_feature, image_feature):
    """计算文本特征与单张图片特征的相似度"""
    text_feature = np.array(text_feature)
    image_feature = np.array(image_feature)
    similarity = np.dot(image_feature, text_feature) / (
        np.linalg.norm(image_feature) * np.linalg.norm(text_feature)
    )
    return similarity


def test_clip_service():
    rospy.init_node("test_clip_node", anonymous=True)
    rospy.wait_for_service("clip")
    clip_service = rospy.ServiceProxy("clip", CLIP)
    cat_img_path = rospy.get_param("~cat_img_path", "/path/to/your/cat_image.jpg")
    dog_img_path = rospy.get_param("~dog_img_path", "/path/to/your/dog_image.jpg")

    # 测试 encode_text
    try:
        text_request = CLIPRequest()
        text_request.mode = "encode_text"
        text_request.text = "A photo of a cat"
        text_response = clip_service(text_request)
        text_feature = text_response.clip_fts
        rospy.loginfo(
            f"Text encoding result: {text_feature[:10]}... (dim: {text_response.clip_ft_dim})"
        )
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call clip service for encode_text: {e}")
        return

    # 测试 encode_image
    try:
        if not os.path.exists(cat_img_path):
            rospy.logerr(f"Image file not found: {cat_img_path}")
            return
        ros_image = create_ros_image_from_file(cat_img_path)

        image_request = CLIPRequest()
        image_request.mode = "encode_image"
        image_request.image = ros_image
        image_response = clip_service(image_request)
        cat_image_feature = image_response.clip_fts
        rospy.loginfo(
            f"Cat image encoding result: {cat_image_feature[:10]}... (dim: {image_response.clip_ft_dim})"
        )
        # 计算文本与小猫图片的相似度
        similarity = calculate_similarity(text_feature, cat_image_feature)
        rospy.loginfo(f"Similarity between text and cat image: {similarity:.4f}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call clip service for encode_image: {e}")

    # 测试 encode_images
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

        images_request = CLIPRequest()
        images_request.mode = "encode_images"
        images_request.images = ros_images
        images_response = clip_service(images_request)
        image_features = np.array(images_response.clip_fts).reshape(
            -1, images_response.clip_ft_dim
        )
        rospy.loginfo(
            f"Images encoding result: {image_features[:10]}... (dim: {images_response.clip_ft_dim})"
        )

        # 计算文本与每张图片的相似度
        similarities = calculate_similarity(text_feature, image_features)
        for i, sim in enumerate(similarities):
            rospy.loginfo(f"Similarity with image {image_paths[i]}: {sim:.4f}")
        most_similar_idx = np.argmax(similarities)
        rospy.loginfo(
            f"The most similar image is: {image_paths[most_similar_idx]} with similarity {similarities[most_similar_idx]:.4f}"
        )
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call clip service for encode_images: {e}")


if __name__ == "__main__":
    try:
        test_clip_service()
    except rospy.ROSInterruptException:
        pass
