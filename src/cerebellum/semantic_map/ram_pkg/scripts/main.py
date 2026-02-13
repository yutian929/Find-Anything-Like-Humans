"""
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
"""
import argparse
import numpy as np
import random
import cv2
import time

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

import os, psutil


def print_mem_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # 常驻内存 (MB)
    print(f"{prefix} Memory usage: {mem:.2f} MB")


parser = argparse.ArgumentParser(
    description="Tag2Text inferece for tagging and captioning"
)
parser.add_argument("--image", metavar="DIR", help="path to dataset", default="328.png")
parser.add_argument(
    "--pretrained",
    metavar="DIR",
    help="path to pretrained model",
    default="weights/ram_plus_swin_large_14m.pth",
)
parser.add_argument(
    "--image-size",
    default=384,
    type=int,
    metavar="N",
    help="input image size (default: 448)",
)


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(
        pretrained=args.pretrained, image_size=args.image_size, vit="swin_l"
    )
    model.eval()
    model = model.to(device)

    # Read video and process frame by frame
    video_capture = cv2.VideoCapture("mc.mp4")
    frame_idx = 0
    while True:
        ret, video_frame = video_capture.read()
        if not ret:
            break

        # cv2.imshow("Video Frame", video_frame)
        # cv2.waitKey(0)
        breakpoint()
        image = transform(Image.fromarray(video_frame)).unsqueeze(0).to(device)
        res = inference(image, model)
        print_mem_usage(f"Frame {frame_idx}:")
        en_list = res[0].split(" | ")
        print(f"Frame {frame_idx} Image Tags: {res[0]}")
        print(f"Frame {frame_idx} 图像标签: {res[1]}")
        print(f"Frame {frame_idx} Image Tags List: {en_list}")
        frame_idx += 1
    video_capture.release()
