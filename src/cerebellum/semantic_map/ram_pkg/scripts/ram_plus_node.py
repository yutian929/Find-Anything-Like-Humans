import rospy
from sensor_msgs.msg import Image
from ram_pkg.srv import RAMPlus, RAMPlusResponse
from cv_bridge import CvBridge
import socket
import struct
import json
import cv2
import time
import numpy as np


def sendall(sock, data: bytes):
    view = memoryview(data)
    while len(view):
        n = sock.send(view)
        view = view[n:]


def recvall(sock, n: int) -> bytes:
    """严格接收 n 字节"""
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


class RAMPlusNode:
    def __init__(self):
        rospy.init_node("ram_plus_node")

        # Socket 服务配置
        self.server_host = rospy.get_param("~server_host", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 55555))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        self.socket_timeout = float(rospy.get_param("~socket_timeout", 5.0))
        self.reconnect_on_each_call = bool(
            rospy.get_param("~reconnect_on_each_call", False)
        )
        # 为了健壮性，默认每次调用都重连；如需长连可设为 False
        self.sock = None

        self.cv_bridge = CvBridge()

        # 等待 worker 可用
        self._wait_for_worker()
        rospy.loginfo(
            f"[ram_plus_node] Ready. Worker at {self.server_host}:{self.server_port}"
        )

        # 广播 ROS Service
        rospy.Service("ram_plus", RAMPlus, self.handle_inference)
        # rospy.loginfo("[ram_plus_node] Service 'ram_plus' is running.")

    def _connect(self):
        """建立 socket 连接"""
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
        """阻塞，直到 worker 可用"""
        # rospy.loginfo(
        #     f"[ram_plus_node] Waiting for worker at {self.server_host}:{self.server_port} ..."
        # )
        while not rospy.is_shutdown():
            try:
                self._connect()
                rospy.loginfo("[ram_plus_node] Connected to worker.")
                self._close()  # 连接成功一次就好，等实际调用时再重连
                return
            except Exception as e:
                rospy.logwarn(f"[ram_plus_node] Worker not available yet: {e}")
                time.sleep(2.0)  # 每 2 秒重试

    def _request_inference(self, bgr_img: np.ndarray):
        """
        通过 socket 把图像发给推理端，并接收 JSON 结果
        协议：
          -> [8字节无符号大端整数: JPEG字节长度] + [JPEG字节]
          <- [8字节无符号大端整数: JSON字节长度] + [JSON字节(utf-8)]
        """
        # 编码为JPEG，降低传输体积
        ok, enc = cv2.imencode(
            ".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg') failed")
        jpg_bytes = enc.tobytes()

        # 连接（如开启每次重连，就每次都重建连接）
        if self.reconnect_on_each_call or self.sock is None:
            self._close()
            self._connect()

        try:
            # 发送：长度前缀 + JPEG
            header = struct.pack("!Q", len(jpg_bytes))  # 8字节无符号大端
            sendall(self.sock, header)
            sendall(self.sock, jpg_bytes)

            # 接收：长度前缀 + JSON
            resp_len_bytes = recvall(self.sock, 8)
            (resp_len,) = struct.unpack("!Q", resp_len_bytes)
            resp_json_bytes = recvall(self.sock, resp_len)
            resp = json.loads(resp_json_bytes.decode("utf-8"))
            return resp
        except Exception:
            # 出错时清理连接，方便下次重连
            self._close()
            raise

    def handle_inference(self, req):
        try:
            # ROS Image -> OpenCV BGR
            cv_img = self.cv_bridge.imgmsg_to_cv2(req.color_image, "bgr8")

            # 发给推理端
            result = self._request_inference(cv_img)

            # 解析推理端返回的 JSON
            tags_en = result.get("tags_en", "")
            tags_cn = result.get("tags_cn", "")
            en_list = result.get("en_list", [])

            # 构造 ROS 响应
            resp = RAMPlusResponse()
            resp.tags_en = tags_en
            resp.tags_cn = tags_cn
            resp.en_list = en_list
            return resp

        except Exception as e:
            rospy.logerr(f"[ram_plus_node] Inference via worker failed: {e}")
            return RAMPlusResponse(tags_en="", tags_cn="", en_list=[])


if __name__ == "__main__":
    try:
        node = RAMPlusNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


# import rospy
# from sensor_msgs.msg import Image
# from ram_pkg.srv import RAMPlus, RAMPlusResponse  # You need to define this service
# from cv_bridge import CvBridge
# import torch
# from PIL import Image as PILImage
# from ram.models import ram_plus
# from ram import inference_ram as inference
# from ram import get_transform
# import numpy as np
# import os, psutil
# import cv2

# def print_mem_usage(prefix=""):
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / 1024**2  # 常驻内存 (MB)
#     print(f"{prefix} Memory usage: {mem:.2f} MB")

# class RAMPlusNode:
#     def __init__(self):
#         rospy.init_node("ram_plus_node")
#         self.cv_bridge = CvBridge()
#         self.image_size = rospy.get_param("~image_size", 384)
#         self.pretrained = rospy.get_param("~pretrained", "weights/ram_plus_swin_large_14m.pth")
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.transform = get_transform(image_size=self.image_size)
#         self.model = ram_plus(pretrained=self.pretrained, image_size=self.image_size, vit='swin_l')
#         self.model.eval()
#         self.model = self.model.to(self.device)
#         rospy.loginfo("RAM++ model loaded.")
#         rospy.Service("ram_plus", RAMPlus, self.handle_inference)
#         rospy.loginfo("ram_plus_node initialized.")

#     def handle_inference(self, req):
#         print_mem_usage("ram_plus_node")
#         try:
#             # Convert ROS Image to PIL Image
#             cv_img = self.cv_bridge.imgmsg_to_cv2(req.color_image, "bgr8")
#             pil_img = PILImage.fromarray(cv_img)
#             # cv2.imshow("Input Image", cv_img)
#             # cv2.waitKey(0)
#             # breakpoint()
#             image_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
#             # with torch.no_grad():
#             res = inference(image_tensor, self.model)
#             # torch.cuda.empty_cache()
#             del image_tensor, cv_img, pil_img
#             tags_en = res[0]
#             tags_cn = res[1]
#             en_list = tags_en.split(" | ")
#             response = RAMPlusResponse()
#             response.tags_en = tags_en
#             response.tags_cn = tags_cn
#             response.en_list = en_list
#             return response
#         except Exception as e:
#             rospy.logerr(f"RAM++ inference error: {e}")
#             return RAMPlusResponse(tags_en="", tags_cn="", en_list=[])


# if __name__ == "__main__":
#     try:
#         node = RAMPlusNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass


# import rospy
# from sensor_msgs.msg import Image
# from ram_pkg.srv import RAMPlus, RAMPlusResponse
# from cv_bridge import CvBridge
# import socket
# import json
# import base64
# import numpy as np
# import cv2

# class RAMPlusNode:
#     def __init__(self):
#         rospy.init_node("ram_plus_node")

#         # 获取参数
#         self.host = rospy.get_param("~host", "localhost")
#         self.port = rospy.get_param("~port", 9999)
#         self.timeout = rospy.get_param("~timeout", 30.0)

#         self.cv_bridge = CvBridge()

#         # 初始化socket连接
#         self.socket = None
#         self.connect_to_inference_server()

#         # 启动服务
#         rospy.Service("ram_plus", RAMPlus, self.handle_inference)
#         rospy.loginfo("ram_plus_node initialized and connected to inference server.")

#         # 设置关闭时的清理函数
#         rospy.on_shutdown(self.cleanup)

#     def connect_to_inference_server(self):
#         """连接到推理服务器"""
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.settimeout(self.timeout)
#             self.socket.connect((self.host, self.port))
#             rospy.loginfo(f"Connected to inference server at {self.host}:{self.port}")
#         except Exception as e:
#             rospy.logerr(f"Failed to connect to inference server: {e}")
#             self.socket = None

#     def handle_inference(self, req):
#         """处理推理请求"""
#         if self.socket is None:
#             rospy.logwarn("Not connected to inference server, attempting to reconnect...")
#             self.connect_to_inference_server()
#             if self.socket is None:
#                 return RAMPlusResponse(tags_en="", tags_cn="", en_list=[])

#         try:
#             # 转换图像格式
#             cv_img = self.cv_bridge.imgmsg_to_cv2(req.color_image, "bgr8")

#             # 编码图像为JPEG格式并转换为base64字符串
#             _, img_encoded = cv2.imencode('.jpg', cv_img)
#             img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

#             # 准备发送的数据
#             data_to_send = {
#                 "image": img_base64,
#                 "image_format": "jpeg"
#             }

#             # 发送数据到推理服务器
#             json_data = json.dumps(data_to_send) + "\n"
#             self.socket.sendall(json_data.encode('utf-8'))

#             # 接收响应
#             response_data = b""
#             while True:
#                 chunk = self.socket.recv(4096)
#                 if not chunk:
#                     break
#                 response_data += chunk
#                 if b"\n" in chunk:
#                     break

#             # 解析响应
#             response_json = json.loads(response_data.decode('utf-8').strip())

#             # 构建ROS响应
#             # breakpoint()
#             response = RAMPlusResponse()
#             response.tags_en = response_json.get("tags_en", "")
#             response.tags_cn = response_json.get("tags_cn", "")
#             response.en_list = response_json.get("en_list", [])

#             return response

#         except socket.timeout:
#             rospy.logerr("Socket timeout while communicating with inference server")
#             return RAMPlusResponse(tags_en="", tags_cn="", en_list=[])
#         except Exception as e:
#             rospy.logerr(f"Error during inference: {e}")
#             # 尝试重新连接
#             self.connect_to_inference_server()
#             return RAMPlusResponse(tags_en="", tags_cn="", en_list=[])

#     def cleanup(self):
#         """清理资源"""
#         if self.socket:
#             self.socket.close()
#             rospy.loginfo("Socket connection closed.")

# if __name__ == "__main__":
#     try:
#         node = RAMPlusNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
