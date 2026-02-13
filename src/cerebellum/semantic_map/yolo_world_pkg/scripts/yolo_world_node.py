import rospy
from cv_bridge import CvBridge
from yolo_world_pkg.srv import YoloDetection, YoloDetectionResponse
from std_msgs.msg import MultiArrayDimension

import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO


class YoloWorldNode:
    def __init__(self, model_path, box_threshold=0.1):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            rospy.logerr("No GPU available")
            exit()

        # rospy.loginfo("Loading YOLO-World model...")
        # Building YOLO-World inference model
        self.yolo_model = YOLO(model_path, task="detect")
        self.box_threshold = box_threshold
        # rospy.loginfo(f"YOLO-World model {model_path} loaded")

        # ROS service
        self.cv_bridge = CvBridge()
        rospy.Service("yolo_detection", YoloDetection, self.callback)
        # rospy.loginfo("yolo_detection service has started")
        # rospy.loginfo("yolo_world_node initialized")

    def detect(self, image, class_list):
        # Detect objects
        self.yolo_model.set_classes(class_list)
        results = self.yolo_model.predict(
            source=image, conf=self.box_threshold, verbose=False
        )
        detections = sv.Detections.from_ultralytics(results[0])

        labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        # Combine labels with scores
        labels_with_scores = [
            f"{results[0].names[int(cls)]} ({conf:.2f})"
            for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf)
        ]

        # Annotate image with detections
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels_with_scores
        )

        return detections, labels, annotated_image, results[0].boxes.xyxy.cpu().numpy()

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")

        class_list = prompt

        detections, labels, annotated_frame, xyxy = self.detect(img, class_list)
        boxes = detections.xyxy
        scores = detections.confidence
        response = YoloDetectionResponse()
        response.labels = labels
        # response.class_id = detections.class_id
        response.scores = scores.tolist()
        response.boxes = boxes.flatten().astype(np.int32).tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame, "bgr8")

        # DEBUG
        # cv2.imshow(f"YOLO-World Detection - {prompt}", annotated_frame)
        # cv2.waitKey(0)

        # Free GPU memory
        torch.cuda.empty_cache()
        return response


if __name__ == "__main__":
    rospy.init_node("yolo_world_node")

    # Get parameters from the ROS parameter server
    model_path = rospy.get_param("~model_path", "weights/yolov8l-worldv2.pt")
    box_threshold = rospy.get_param("~box_threshold", 0.1)

    # Start the node
    YoloWorldNode(model_path, box_threshold)
    rospy.spin()
