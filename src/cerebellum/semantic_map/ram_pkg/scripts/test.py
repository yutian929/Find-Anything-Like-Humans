import rospy
import cv2
import numpy as np
from ram_pkg.srv import RAMPlus, RAMPlusRequest
from cv_bridge import CvBridge


def test_ram_plus_video(video_path):
    rospy.wait_for_service("ram_plus")
    ram_plus_srv = rospy.ServiceProxy("ram_plus", RAMPlus)
    bridge = CvBridge()
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    while not rospy.is_shutdown():
        ret, frame = video_capture.read()
        if not ret:
            break
        ros_img = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        req = RAMPlusRequest(color_image=ros_img)
        try:
            # breakpoint()
            res = ram_plus_srv(req)
            en_list = res.tags_en.split(" | ")
            print(f"Frame {frame_idx} Image Tags: {res.tags_en}")
            print(f"Frame {frame_idx} 图像标签: {res.tags_cn}")
            print(f"Frame {frame_idx} Image Tags List: {en_list}")
        except rospy.ServiceException as e:
            rospy.logerr(f"RAMPlus service call failed: {e}")
        frame_idx += 1
    video_capture.release()


if __name__ == "__main__":
    rospy.init_node("test_ram_plus_node")
    video_path = "/home/yutian/YanBot/src/cerebellum/semantic_map/ram_pkg/scripts/mc.mp4"  # Replace with the path to your test video
    test_ram_plus_video(video_path)
