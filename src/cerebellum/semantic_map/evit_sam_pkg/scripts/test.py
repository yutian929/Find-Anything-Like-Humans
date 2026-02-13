import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
from evit_sam_pkg.srv import EvitSamSegmentation, EvitSamSegmentationRequest


def build_boxes_array(boxes):
    """
    boxes: list of [x1, y1, x2, y2] (N x 4)
    """
    msg = Float32MultiArray()
    data = np.array(boxes, dtype=np.float32).flatten()
    msg.data = data.tolist()

    dim = MultiArrayDimension()
    dim.label = "boxes"
    dim.size = len(boxes)
    dim.stride = len(data)
    msg.layout.dim = [dim]
    return msg


def main():
    rospy.init_node("test_sam_segmentation")
    rospy.wait_for_service("sam_segmentation")
    bridge = CvBridge()

    try:
        service = rospy.ServiceProxy("sam_segmentation", EvitSamSegmentation)

        # === 加载图片（修改为你自己的路径） ===
        image_path = "/home/yutian/YanBot/src/cerebellum/semantic_map/evit_sam_pkg/scripts/328.png"
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (640, 480))  # 保证尺寸统一
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_msg = bridge.cv2_to_imgmsg(img_resized, encoding="bgr8")

        # === 设置请求 ===
        request = EvitSamSegmentationRequest()
        request.model = "fast"  # or "fast"
        request.mode = "bbox"  # or "full"
        request.color_image = img_msg

        # 如果是bbox模式，设置boxes
        if request.mode == "bbox":
            # 示例: 两个框
            boxes = [[100, 100, 200, 250], [300, 150, 400, 300]]
            request.boxes = build_boxes_array(boxes)
        else:
            request.boxes = build_boxes_array([])

        # === 调用服务 ===
        response = service(request)

        # === 显示结果 ===
        print(f"Received {len(response.seg_masks)} masks")

        for i, mask_msg in enumerate(response.seg_masks):
            mask = bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
            cv2.imshow(f"Mask {i}", mask)

        masked_frame = bridge.imgmsg_to_cv2(
            response.masked_frame, desired_encoding="bgr8"
        )
        cv2.imshow("Masked Image", masked_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    main()
