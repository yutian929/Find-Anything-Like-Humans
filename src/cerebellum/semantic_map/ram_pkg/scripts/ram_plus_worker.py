import socket
import struct
import json
import threading
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


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


class RAMPlusWorker:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 55555,
        image_size: int = 384,
        pretrained_path: str = "weights/ram_plus_swin_large_14m.pth",
        vit: str = "swin_l",
        device: str = None,
        num_threads: int = 0,
    ):
        self.host = host
        self.port = int(port)
        self.image_size = int(image_size)
        self.pretrained = pretrained_path
        self.vit = vit

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)

        # 预加载模型与变换
        print(f"[ram_plus_worker] Loading model on {self.device} ...")
        self.transform = get_transform(image_size=self.image_size)
        self.model = ram_plus(
            pretrained=self.pretrained, image_size=self.image_size, vit=self.vit
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        print("[ram_plus_worker] Model loaded.")

        # 建立监听 socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(8)
        print(f"[ram_plus_worker] Listening on {self.host}:{self.port}")

    def _handle_conn(self, conn: socket.socket, addr):
        try:
            while True:
                # 读取一帧请求（图像）
                header = conn.recv(8)
                if not header:
                    break
                (img_len,) = struct.unpack("!Q", header)
                img_bytes = recvall(conn, img_len)

                # 解码JPEG -> BGR
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("cv2.imdecode failed")

                # BGR -> RGB 再转 PIL（更稳妥）
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)

                # 预处理 + 推理
                with torch.no_grad():
                    image_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    res = inference(image_tensor, self.model)

                # 释放中间变量
                del image_tensor, bgr, rgb, pil_img

                # 整理输出
                tags_en = (
                    res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else ""
                )
                tags_cn = (
                    res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else ""
                )
                en_list = tags_en.split(" | ") if tags_en else []

                payload = {
                    "tags_en": tags_en,
                    "tags_cn": tags_cn,
                    "en_list": en_list,
                }
                out = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                sendall(conn, struct.pack("!Q", len(out)))
                sendall(conn, out)

        except Exception as e:
            print(f"[ram_plus_worker] Error handling {addr}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve_forever(self):
        while True:
            conn, addr = self.server.accept()
            # 每个连接启个线程，保证并发与互不影响
            t = threading.Thread(
                target=self._handle_conn, args=(conn, addr), daemon=True
            )
            t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAM++ Socket Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55555)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument(
        "--pretrained",
        default=os.path.join(
            os.path.dirname(__file__), "weights", "ram_plus_swin_large_14m.pth"
        ),
    )
    parser.add_argument("--vit", default="swin_l")
    parser.add_argument("--device", default="cuda", help="cuda / cpu (default: auto)")
    parser.add_argument(
        "--num_threads", type=int, default=0, help="intra-op threads for torch"
    )

    # 👇 加上这一行，忽略 ROS 传过来的 __name:=xxx __log:=xxx
    args, unknown = parser.parse_known_args(sys.argv[1:])

    worker = RAMPlusWorker(
        host=args.host,
        port=args.port,
        image_size=args.image_size,
        pretrained_path=args.pretrained,
        vit=args.vit,
        device=args.device,
        num_threads=args.num_threads,
    )
    worker.serve_forever()


# #!/usr/bin/env python

# import socket
# import json
# import base64
# import numpy as np
# from PIL import Image as PILImage
# import io
# import torch
# from ram.models import ram_plus
# from ram import inference_ram as inference
# from ram import get_transform

# class RAMPlusInferenceServer:
#     def __init__(self, host='localhost', port=9999, image_size=384,
#                  pretrained="weights/ram_plus_swin_large_14m.pth"):
#         self.host = host
#         self.port = port
#         self.image_size = image_size

#         # 设置设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # 加载模型
#         self.transform = get_transform(image_size=self.image_size)
#         self.model = ram_plus(pretrained=pretrained, image_size=self.image_size, vit='swin_l')
#         self.model.eval()
#         self.model = self.model.to(self.device)

#         # 创建socket服务器
#         self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         self.socket.bind((self.host, self.port))
#         self.socket.listen(1)

#         print(f"RAM++ Inference Server listening on {self.host}:{self.port}")
#         print(f"Model loaded with image size: {image_size}")

#     def base64_to_pil_image(self, base64_str, image_format="jpeg"):
#         """将base64字符串转换为PIL图像"""
#         img_data = base64.b64decode(base64_str)
#         img_buffer = io.BytesIO(img_data)
#         return PILImage.open(img_buffer)

#     def process_image(self, image_base64):
#         """处理图像并返回推理结果"""
#         try:
#             # 转换图像
#             pil_img = self.base64_to_pil_image(image_base64)

#             # 应用变换并推理
#             image_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 res = inference(image_tensor, self.model)

#             # 清理GPU内存
#             del image_tensor
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#             # 准备结果
#             tags_en = res[0]
#             tags_cn = res[1]
#             en_list = tags_en.split(" | ")

#             return {
#                 "tags_en": tags_en,
#                 "tags_cn": tags_cn,
#                 "en_list": en_list
#             }

#         except Exception as e:
#             print(f"Error during inference: {e}")
#             return {
#                 "tags_en": "",
#                 "tags_cn": "",
#                 "en_list": []
#             }

#     def run(self):
#         """运行服务器主循环"""
#         try:
#             while True:
#                 print("Waiting for connection...")
#                 client_socket, addr = self.socket.accept()
#                 print(f"Connected by {addr}")

#                 try:
#                     # 接收数据
#                     data = b""
#                     while True:
#                         chunk = client_socket.recv(4096)
#                         if not chunk:
#                             break
#                         data += chunk
#                         if b"\n" in chunk:
#                             break

#                     if not data:
#                         continue

#                     # 解析JSON数据
#                     request_data = json.loads(data.decode('utf-8').strip())
#                     image_base64 = request_data.get("image", "")

#                     if image_base64:
#                         # 处理图像
#                         result = self.process_image(image_base64)

#                         # 发送响应
#                         response_json = json.dumps(result) + "\n"
#                         client_socket.sendall(response_json.encode('utf-8'))
#                     else:
#                         error_response = json.dumps({
#                             "error": "No image data provided"
#                         }) + "\n"
#                         client_socket.sendall(error_response.encode('utf-8'))

#                 except Exception as e:
#                     print(f"Error handling client: {e}")
#                     error_response = json.dumps({
#                         "error": str(e)
#                     }) + "\n"
#                     client_socket.sendall(error_response.encode('utf-8'))

#                 finally:
#                     client_socket.close()

#         except KeyboardInterrupt:
#             print("Server shutting down...")
#         finally:
#             self.socket.close()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description='RAM++ Inference Server')
#     parser.add_argument('--host', default='localhost', help='Host address')
#     parser.add_argument('--port', type=int, default=9999, help='Port number')
#     parser.add_argument('--image_size', type=int, default=384, help='Image size for model')
#     parser.add_argument('--pretrained', default='weights/ram_plus_swin_large_14m.pth',
#                         help='Path to pretrained model weights')

#     args = parser.parse_args()

#     server = RAMPlusInferenceServer(
#         host=args.host,
#         port=args.port,
#         image_size=args.image_size,
#         pretrained=args.pretrained
#     )

#     server.run()
