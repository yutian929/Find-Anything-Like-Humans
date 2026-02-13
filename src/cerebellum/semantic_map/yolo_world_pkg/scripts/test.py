import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolo_world_pkg.srv import YoloDetection
import numpy as np


class YoloWorldTestNode:
    def __init__(self):
        rospy.init_node("yolo_world_test_node")
        self.cv_bridge = CvBridge()
        self.image_received = False
        self.test_classes = ["person", "cup", "chair"]  # Test classes to detect

        # Wait for the service to become available
        rospy.loginfo("Waiting for yolo_detection service...")
        rospy.wait_for_service("yolo_detection")
        self.yolo_client = rospy.ServiceProxy("yolo_detection", YoloDetection)
        rospy.loginfo("Connected to yolo_detection service")

        # Subscribe to the camera topic
        # self.image_sub = rospy.Subscriber(
        #     "/camera/color/image_raw", Image, self.image_callback, queue_size=1
        # )
        # rospy.loginfo("Subscribed to /camera/color/image_raw")

        self.image_sub = rospy.Subscriber(
            "/ai2thor/rgb", Image, self.image_callback, queue_size=1
        )
        rospy.loginfo("Subscribed to /ai2thor/rgb")

        # Publisher for annotated images
        self.result_pub = rospy.Publisher(
            "/yolo_test/annotated_image", Image, queue_size=1
        )

    def image_callback(self, msg):
        if not self.image_received:
            self.image_received = True
            rospy.loginfo("First image received, running test...")

            try:
                # Convert ROS Image to OpenCV format
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

                # Create test prompt (comma-separated class list)
                prompt = self.test_classes
                rospy.loginfo(f"Testing with prompt: {prompt}")

                # Call the YoloWorld service
                start_time = rospy.Time.now()
                response = self.yolo_client(msg, prompt)
                elapsed = (rospy.Time.now() - start_time).to_sec()

                # Process the results
                labels = response.labels
                scores = response.scores

                # Log the detection results
                if len(labels) > 0:
                    rospy.loginfo(
                        f"Test PASSED! Detected {len(labels)} objects in {elapsed:.2f}s"
                    )
                    for i, (label, score) in enumerate(zip(labels, scores)):
                        rospy.loginfo(f"  {i+1}. {label}: {score:.2f}")
                else:
                    rospy.loginfo(
                        f"Test completed. No objects detected from classes: {self.test_classes}"
                    )

                # Publish the annotated image
                self.result_pub.publish(response.annotated_frame)
                rospy.loginfo("Published annotated image to /yolo_test/annotated_image")

            except Exception as e:
                rospy.logerr(f"Test FAILED: {str(e)}")

            # Test on subsequent images too
            self.image_received = False


if __name__ == "__main__":
    try:
        test_node = YoloWorldTestNode()
        rospy.loginfo("YoloWorld Test Node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
