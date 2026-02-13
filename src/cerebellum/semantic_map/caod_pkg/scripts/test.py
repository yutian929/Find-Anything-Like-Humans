import rospy
import cv2
import numpy as np
from caod_pkg.srv import CAOD, CAODRequest
from cv_bridge import CvBridge


def draw_bboxes(image, bboxes, scores):
    """
    Draw bounding boxes and scores on the image.
    :param image: OpenCV image
    :param bboxes: List of bounding boxes [x1, y1, x2, y2]
    :param scores: List of scores corresponding to the bounding boxes
    """
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def test_mvits_node(image_path):
    rospy.wait_for_service("caod")
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            rospy.logerr(f"Failed to read image from {image_path}")
            return

        # Convert image to ROS Image message
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(image, encoding="bgr8")

        # Create service proxy
        caod_service = rospy.ServiceProxy("caod", CAOD)

        # Test mode "nms" with varying nms_threshold
        for nms_threshold in np.arange(
            0.0, 1.1, 0.1
        ):  # From 0.0 to 1.0 in steps of 0.1
            req = CAODRequest()
            req.color_image = ros_image
            req.mode = "nms"
            req.score_threshold = 0.1
            req.nms_threshold = nms_threshold
            rospy.loginfo(
                f"Calling service with mode 'nms', nms_threshold={nms_threshold}"
            )
            res = caod_service(req)
            if res.scores and res.boxes:
                bboxes = np.array(res.boxes).reshape(-1, 4)
                scores = np.array(res.scores)
                image_nms = image.copy()
                draw_bboxes(image_nms, bboxes, scores)
                cv2.imshow(f"Mode: nms, nms_threshold={nms_threshold:.1f}", image_nms)
                cv2.waitKey(0)

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    rospy.init_node("test_mvits_node")
    image_path = "/home/yutian/YanBot/src/cerebellum/semantic_map/caod_pkg/scripts/test_dir/A328.jpg"  # Replace with the path to your test image
    test_mvits_node(image_path)
    exit(0)
