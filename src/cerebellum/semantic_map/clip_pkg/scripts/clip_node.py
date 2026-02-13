import rospy
from clip_pkg.srv import CLIP, CLIPResponse
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


class CLIPNode:
    def __init__(self):
        rospy.init_node("clip_node")

        self.server_host = rospy.get_param("~server_host", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 55557))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        self.socket_timeout = float(rospy.get_param("~socket_timeout", 5.0))
        self.reconnect_on_each_call = bool(
            rospy.get_param("~reconnect_on_each_call", False)
        )

        self.sock = None
        self.bridge = CvBridge()

        self._wait_for_worker()
        rospy.loginfo(
            f"[clip_node] Ready. Worker at {self.server_host}:{self.server_port}"
        )

        rospy.Service("clip", CLIP, self.handle_clip_request)

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
                rospy.loginfo("[clip_node] Connected to worker.")
                self._close()
                return
            except Exception as e:
                rospy.logwarn(f"[clip_node] Worker not available yet: {e}")
                time.sleep(2.0)

    def _request_inference(self, mode, text=None, images=None):
        if self.reconnect_on_each_call or self.sock is None:
            self._close()
            self._connect()
        try:
            # Prepare payload
            payload = {"mode": mode}
            if text is not None:
                payload["text"] = text
            if images is not None:
                # images: list of np.ndarray (BGR)
                jpg_list = []
                for img in images:
                    ok, enc = cv2.imencode(
                        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    )
                    if not ok:
                        raise RuntimeError("cv2.imencode('.jpg') failed")
                    jpg_list.append(enc.tobytes())
                payload["images"] = [jpg.hex() for jpg in jpg_list]
            # Send JSON
            payload_bytes = json.dumps(payload).encode("utf-8")
            header = struct.pack("!Q", len(payload_bytes))
            sendall(self.sock, header)
            sendall(self.sock, payload_bytes)
            # Receive response
            resp_len_bytes = recvall(self.sock, 8)
            (resp_len,) = struct.unpack("!Q", resp_len_bytes)
            resp_json_bytes = recvall(self.sock, resp_len)
            resp = json.loads(resp_json_bytes.decode("utf-8"))
            return resp
        except Exception:
            self._close()
            raise

    def handle_clip_request(self, req):
        try:
            mode = req.mode
            if mode == "encode_text":
                resp = self._request_inference(mode, text=req.text)
                clip_fts = resp.get("clip_fts", [])
                clip_ft_dim = resp.get("clip_ft_dim", 0)
                return CLIPResponse(clip_fts=clip_fts, clip_ft_dim=clip_ft_dim)
            elif mode == "encode_image":
                cv_img = self.bridge.imgmsg_to_cv2(req.image, "rgb8")
                resp = self._request_inference(mode, images=[cv_img])
                clip_fts = resp.get("clip_fts", [])
                clip_ft_dim = resp.get("clip_ft_dim", 0)
                return CLIPResponse(clip_fts=clip_fts, clip_ft_dim=clip_ft_dim)
            elif mode == "encode_images":
                cv_imgs = [self.bridge.imgmsg_to_cv2(img, "rgb8") for img in req.images]
                resp = self._request_inference(mode, images=cv_imgs)
                clip_fts = resp.get("clip_fts", [])
                clip_ft_dim = resp.get("clip_ft_dim", 0)
                return CLIPResponse(clip_fts=clip_fts, clip_ft_dim=clip_ft_dim)
            else:
                rospy.logerr(f"[clip_node] Unknown mode: {mode}")
                return CLIPResponse(clip_fts=[], clip_ft_dim=0)
        except Exception as e:
            rospy.logerr(f"[clip_node] Inference via worker failed: {e}")
            return CLIPResponse(clip_fts=[], clip_ft_dim=0)


if __name__ == "__main__":
    try:
        node = CLIPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
if __name__ == "__main__":
    try:
        node = CLIPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
