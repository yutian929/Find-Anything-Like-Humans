import socket
import struct
import json
import threading
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from models.model import Model  # 与原 mvits_node 一致


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


class MVITSWorker:
    def __init__(
        self,
        host="0.0.0.0",
        port=55556,
        model_type="mdef_detr_minus_language",
        checkpoints_path="MDef_DETR_minus_language_r101_epoch10.pth",
        device=None,
        num_threads=0,
    ):
        self.host = host
        self.port = int(port)
        self.model_type = model_type
        self.checkpoints_path = checkpoints_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[mvits_worker] Loading model ({self.model_type}) on {self.device} ...")
        self.model = Model(self.model_type, self.checkpoints_path).get_model()

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(8)
        print(f"[mvits_worker] Listening on {self.host}:{self.port}")

    def _infer_bgr(self, bgr: np.ndarray):
        """
        与原 mvits_node.run_inference 保持一致：
        将图像写临时文件 -> self.model.infer_image(img_path, caption='all objects')
        返回 (boxes, scores)，均为 Python list
        """
        # 更稳的路径（避免竞争）
        tmp_path = "/tmp/caod_worker.jpg"
        cv2.imwrite(tmp_path, bgr)

        # 与原实现保持调用方式一致
        boxes, scores = self.model.infer_image(tmp_path, caption="all objects")

        # 释放显存的策略按需开启
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 确保为可 JSON 序列化
        boxes = [[int(x) for x in box] for box in boxes]
        scores = [float(s) for s in scores]
        return boxes, scores

    def _handle_conn(self, conn: socket.socket, addr):
        try:
            while True:
                header = conn.recv(8)
                if not header:
                    break
                (img_len,) = struct.unpack("!Q", header)
                img_bytes = recvall(conn, img_len)

                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("cv2.imdecode failed")

                boxes, scores = self._infer_bgr(bgr)

                payload = {"boxes": boxes, "scores": scores}
                out = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                sendall(conn, struct.pack("!Q", len(out)))
                sendall(conn, out)

        except Exception as e:
            print(f"[mvits_worker] Error handling {addr}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve_forever(self):
        while True:
            conn, addr = self.server.accept()
            t = threading.Thread(
                target=self._handle_conn, args=(conn, addr), daemon=True
            )
            t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MViTs Socket Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55556)
    parser.add_argument("--model_type", default="mdef_detr_minus_language")
    parser.add_argument(
        "--model_checkpoints_path", default="MDef_DETR_minus_language_r101_epoch10.pth"
    )
    parser.add_argument("--device", default="cuda")  # cuda / cpu / None(auto)
    parser.add_argument("--num_threads", type=int, default=0)
    args, unknown = parser.parse_known_args()

    worker = MVITSWorker(
        host=args.host,
        port=args.port,
        model_type=args.model_type,
        checkpoints_path=args.model_checkpoints_path,
        device=args.device,
        num_threads=args.num_threads,
    )
    worker.serve_forever()
