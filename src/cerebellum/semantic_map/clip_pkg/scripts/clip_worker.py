import socket
import struct
import json
import threading
import argparse
import os
import cv2
import numpy as np
import torch
import clip
from PIL import Image as PILImage


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


class CLIPWorker:
    def __init__(
        self,
        host="0.0.0.0",
        port=55557,
        model_type="ViT-B/16",
        device=None,
        num_threads=0,
    ):
        self.host = host
        self.port = int(port)
        self.model_type = model_type

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[clip_worker] Loading model ({self.model_type}) on {self.device} ...")
        self.model, self.preprocess = clip.load(self.model_type, device=self.device)
        print(f"[clip_worker] Model loaded.")

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(8)
        print(f"[clip_worker] Listening on {self.host}:{self.port}")

    def _encode_text(self, text):
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        torch.cuda.empty_cache()
        return text_features.cpu().numpy()

    def _encode_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(img_rgb)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        torch.cuda.empty_cache()
        return image_features.cpu().numpy()

    def _encode_images(self, imgs_bgr):
        images = []
        for img_bgr in imgs_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(img_rgb)
            image = self.preprocess(pil_image)
            images.append(image)
        if not images:
            return None
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            batch_features = self.model.encode_image(batch)
        torch.cuda.empty_cache()
        return batch_features.cpu().numpy()

    def _handle_conn(self, conn: socket.socket, addr):
        try:
            while True:
                header = conn.recv(8)
                if not header:
                    break
                (payload_len,) = struct.unpack("!Q", header)
                payload_bytes = recvall(conn, payload_len)
                payload = json.loads(payload_bytes.decode("utf-8"))

                mode = payload.get("mode")
                if mode == "encode_text":
                    text = payload.get("text", "")
                    features = self._encode_text(text)
                    features = features.flatten().astype(float).tolist()
                    ft_dim = (
                        features.__len__()
                        if isinstance(features, list)
                        else features.shape[-1]
                    )
                    resp = {"clip_fts": features, "clip_ft_dim": ft_dim}
                elif mode == "encode_image":
                    imgs = payload.get("images", [])
                    if not imgs:
                        resp = {"clip_fts": [], "clip_ft_dim": 0}
                    else:
                        img_bgr = cv2.imdecode(
                            np.frombuffer(bytes.fromhex(imgs[0]), np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        features = self._encode_image(img_bgr)
                        features = features.flatten().astype(float).tolist()
                        ft_dim = (
                            features.__len__()
                            if isinstance(features, list)
                            else features.shape[-1]
                        )
                        resp = {"clip_fts": features, "clip_ft_dim": ft_dim}
                elif mode == "encode_images":
                    imgs = payload.get("images", [])
                    if not imgs:
                        resp = {"clip_fts": [], "clip_ft_dim": 0}
                    else:
                        imgs_bgr = [
                            cv2.imdecode(
                                np.frombuffer(bytes.fromhex(j), np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            for j in imgs
                        ]
                        features = self._encode_images(imgs_bgr)
                        features = features.flatten().astype(float).tolist()
                        ft_dim = (
                            features.__len__() // len(imgs_bgr)
                            if len(imgs_bgr) > 0
                            else 0
                        )
                        resp = {"clip_fts": features, "clip_ft_dim": ft_dim}
                else:
                    resp = {"clip_fts": [], "clip_ft_dim": 0}

                out = json.dumps(resp, ensure_ascii=False).encode("utf-8")
                sendall(conn, struct.pack("!Q", len(out)))
                sendall(conn, out)

        except Exception as e:
            print(f"[clip_worker] Error handling {addr}: {e}")
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
    parser = argparse.ArgumentParser(description="CLIP Socket Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55557)
    parser.add_argument("--model_type", default="ViT-B/16")
    parser.add_argument("--device", default="cuda")  # cuda / cpu / None(auto)
    parser.add_argument("--num_threads", type=int, default=0)
    args, unknown = parser.parse_known_args()

    worker = CLIPWorker(
        host=args.host,
        port=args.port,
        model_type=args.model_type,
        device=args.device,
        num_threads=args.num_threads,
    )
    worker.serve_forever()
