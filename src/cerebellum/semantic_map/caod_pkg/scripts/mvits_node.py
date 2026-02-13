import rospy
from caod_pkg.srv import CAOD, CAODResponse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import socket
import struct
import json
import time
import cv2
import numpy as np


def sendall(sock, data: bytes):
    view = memoryview(data)
    while len(view):
        n = sock.send(view)
        view = view[n:]


def recvall(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv(n - got)
        if not r:
            raise ConnectionError("Socket closed during recv")
        ln = len(r)
        view[got : got + ln] = r
        got += ln
    return bytes(buf)


class MVITSNode:
    def __init__(self):
        rospy.init_node("mvits_node")

        # socket/server 参数与 RAM++ 节点一致风格
        self.server_host = rospy.get_param("~server_host", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 55556))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        self.socket_timeout = float(rospy.get_param("~socket_timeout", 5.0))
        self.reconnect_on_each_call = bool(
            rospy.get_param("~reconnect_on_each_call", False)
        )

        self.sock = None
        self.bridge = CvBridge()

        self._wait_for_worker()
        rospy.loginfo(
            f"[mvits_node] Ready. Worker at {self.server_host}:{self.server_port}"
        )

        rospy.Service("mvits", CAOD, self.handle_inference)  # 保持原服务名与接口

    def _connect(self):
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.socket_timeout)
        s.connect((self.server_host, self.server_port))
        self.sock = s

    def _close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def _wait_for_worker(self):
        while not rospy.is_shutdown():
            try:
                self._connect()
                rospy.loginfo("[mvits_node] Connected to worker.")
                self._close()
                return
            except Exception as e:
                rospy.logwarn(f"[mvits_node] Worker not available yet: {e}")
                time.sleep(2.0)

    def _request_inference(self, bgr_img: np.ndarray):
        ok, enc = cv2.imencode(
            ".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg') failed")
        jpg_bytes = enc.tobytes()

        if self.reconnect_on_each_call or self.sock is None:
            self._close()
            self._connect()

        try:
            header = struct.pack("!Q", len(jpg_bytes))
            sendall(self.sock, header)
            sendall(self.sock, jpg_bytes)

            resp_len_bytes = recvall(self.sock, 8)
            (resp_len,) = struct.unpack("!Q", resp_len_bytes)
            resp_json_bytes = recvall(self.sock, resp_len)
            resp = json.loads(resp_json_bytes.decode("utf-8"))
            return resp
        except Exception:
            self._close()
            raise

    @staticmethod
    def nms(boxes, scores, score_threshold, nms_threshold):
        """
        boxes: (N,4) 为 [x1,y1,x2,y2]
        """
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # cv2.dnn.NMSBoxes 接口接受 list；旧实现直接传入 xyxy
        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
        )
        if len(idxs) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        idxs = np.array(idxs).flatten()
        return boxes[idxs], scores[idxs]

    @staticmethod
    def create_annotated_frame(image, boxes, scores):
        img = image.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{float(score):.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return img

    def handle_inference(self, req):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(req.color_image, "bgr8")
            result = self._request_inference(cv_img)

            boxes = np.array(result.get("boxes", []))
            scores = np.array(result.get("scores", []))

            if req.mode == "all":
                use_boxes, use_scores = boxes, scores
            elif req.mode == "nms":
                use_boxes, use_scores = self.nms(
                    boxes, scores, req.score_threshold, req.nms_threshold
                )
            else:
                rospy.logerr(f"[mvits_node] Unknown mode: {req.mode}")
                return CAODResponse()

            annotated = self.create_annotated_frame(cv_img, use_boxes, use_scores)

            res = CAODResponse()
            res.scores = use_scores.tolist()
            res.boxes = use_boxes.flatten().tolist()
            res.annotated_frame = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            return res

        except Exception as e:
            rospy.logerr(f"[mvits_node] Inference via worker failed: {e}")
            return CAODResponse()


if __name__ == "__main__":
    try:
        node = MVITSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


# import gc
# from PIL import Image
# import numpy as np
# import rospy
# from caod_pkg.srv import CAOD, CAODResponse
# from models.model import Model
# from cv_bridge import CvBridge
# import cv2
# import torch

# import os, psutil


# def print_mem_usage(prefix=""):
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / 1024**2  # 常驻内存 (MB)
#     print(f"{prefix} Memory usage: {mem:.2f} MB")


# class MVITSNode:
#     def __init__(self):
#         rospy.init_node("mvits_node")
#         torch.cuda.empty_cache()
#         self.model_type = rospy.get_param("~model_type", "mdef_detr_minus_language")
#         self.checkpoints_path = rospy.get_param(
#             "~model_checkpoints_path", "MDef_DETR_minus_language_r101_epoch10.pth"
#         )
#         self.model = Model(self.model_type, self.checkpoints_path).get_model()
#         self.bridge = CvBridge()
#         rospy.Service("caod", CAOD, self.handle_inference)

#     def nms(self, boxes, scores, score_threshold, nms_threshold):
#         """
#         Perform Non-Maximum Suppression (NMS) on the bounding boxes.
#         :param boxes: numpy array of shape (N, 4) where N is the number of boxes
#         :param scores: numpy array of shape (N,) with scores for each box
#         :param score_threshold: float, score threshold for filtering
#         :param nms_threshold: float, IOU threshold for NMS
#         :return: filtered boxes and scores after applying NMS
#         """
#         indices = cv2.dnn.NMSBoxes(
#             boxes.tolist(),
#             scores.tolist(),
#             score_threshold=score_threshold,
#             nms_threshold=nms_threshold,
#         )
#         if len(indices) == 0:
#             return np.array([]), np.array([])
#         return boxes[indices.flatten()], scores[indices.flatten()]

#     def create_annotated_frame(self, image, boxes, scores):
#         for box, score in zip(boxes, scores):
#             x1, y1, x2, y2 = box.astype(int)
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 image,
#                 f"{score:.2f}",
#                 (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )
#         return image

#     def handle_inference(self, req):
#         print_mem_usage("mvits_node")
#         # print("GPU allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
#         # print("GPU reserved:", torch.cuda.memory_reserved()/1024**2, "MB")
#         # import objgraph
#         # objgraph.show_growth(limit=10)
#         # import gc
#         # print(f"Objects in memory: {len(gc.get_objects())}")

#         try:
#             # Convert ROS Image to OpenCV image
#             cv_image = self.bridge.imgmsg_to_cv2(req.color_image, "bgr8")

#             # Perform inference
#             boxes, scores = self.run_inference(cv_image)

#             res = CAODResponse()
#             if req.mode == "all":
#                 res.scores = scores.tolist()
#                 res.boxes = (
#                     boxes.flatten().tolist()
#                 )  # Flattened storage of bounding boxes
#                 annotated_frame = self.create_annotated_frame(cv_image, boxes, scores)
#             elif req.mode == "nms":
#                 nms_boxes, nms_scores = self.nms(
#                     boxes, scores, req.score_threshold, req.nms_threshold
#                 )
#                 print(f"params: {req.score_threshold}, {req.nms_threshold}")
#                 res.scores = nms_scores.tolist()
#                 res.boxes = (
#                     nms_boxes.flatten().tolist()
#                 )  # Flattened storage of bounding boxes
#                 annotated_frame = self.create_annotated_frame(
#                     cv_image, nms_boxes, nms_scores
#                 )
#             else:
#                 rospy.logerr(f"Unknown mode: {req.mode}")
#                 return CAODResponse()
#             annotated_frame_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
#             res.annotated_frame = annotated_frame_msg
#             return res
#         except Exception as e:
#             rospy.logerr(f"Error during inference: {e}")
#             return CAODResponse()

#     def run_inference(self, image):
#         # new_infer
#         # cv_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # pil_img = Image.fromarray(cv_img_rgb)
#         # with torch.no_grad():  # Ensure no gradients are computed
#         # boxes, scores = self.model.infer_image(pil_img)
#         # torch.cuda.empty_cache()

#         img_path = "/tmp/caod.jpg"
#         cv2.imwrite(img_path, image)
#         boxes, scores = self.model.infer_image(img_path, caption="all objects")

#         return np.array(boxes), np.array(scores)

#     def spin(self):
#         rospy.loginfo("MVITSNode is ready.")
#         rospy.spin()


# if __name__ == "__main__":
#     node = MVITSNode()
#     node.spin()
