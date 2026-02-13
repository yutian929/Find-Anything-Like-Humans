import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tkinter as tk
from tkinter import messagebox
from PIL import Image as PILImage, ImageTk
from clip_pkg.srv import CLIP, CLIPRequest, CLIPResponse
from dinov2_pkg.srv import DINOv2, DINOv2Request, DINOv2Response
from dinov3_pkg.srv import DINOv3, DINOv3Request, DINOv3Response
from preset_database import PresetDB
import threading
import os
import time


class PresetAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("ROS Preset Annotator")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        rospy.init_node("preset_annotator")
        # ROS Parameters
        ## General
        self.cv_bridge = CvBridge()
        self.cv_img = None
        self.orig_img = None  # 存储原始未缩放图像
        self.lock = threading.Lock()
        self.feature_encode_head = rospy.get_param(
            "~feature_encode_head", "dinov3"
        )  # clip, dinov2, dinov3
        ## Input - RGB
        # self.rgb_sub_topic = rospy.get_param("~rgb_sub", "/ai2thor/rgb")
        self.rgb_sub_topic = rospy.get_param("~rgb_sub", "/camera/color/image_raw")
        ## Input - DB
        db_path = rospy.get_param("~db_path", "preset.db")
        renew_db = rospy.get_param("~renew_db", False)

        # Server & Client
        ## feature_encoding
        if self.feature_encode_head == "clip":
            rospy.loginfo("Waiting for clip_pkg clip service...")
            rospy.wait_for_service("clip")
            self.clip_client = rospy.ServiceProxy("clip", CLIP)
        elif self.feature_encode_head == "dinov2":
            rospy.loginfo("Waiting for dinov2_pkg dinov2 service...")
            rospy.wait_for_service("dinov2")
            self.dinov2_client = rospy.ServiceProxy("dinov2", DINOv2)
        elif self.feature_encode_head == "dinov3":
            rospy.loginfo("Waiting for dinov3_pkg dinov3 service...")
            rospy.wait_for_service("dinov3")
            self.dinov3_client = rospy.ServiceProxy("dinov3", DINOv3)
        else:
            rospy.logerr("Unsupportable feature_encode_head !")
            return

        # Subscriber & Publisher
        ## Subscribe to RGB
        self.rgb_sub = rospy.Subscriber(self.rgb_sub_topic, Image, self.rgb_callback)

        # DB
        self.db = PresetDB(db_path=db_path, renew_db=renew_db)

        # 状态控制
        self.freeze = False
        self.rect = None
        self.start_x, self.start_y = 0, 0
        self.scale_x, self.scale_y = 1.0, 1.0  # 图像缩放比例

        # Tkinter 控件
        self.canvas = tk.Canvas(root, width=640, height=480, bg="gray")
        self.canvas.pack()

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, pady=5)

        self.start_btn = tk.Button(button_frame, text="开始认定", command=self.start_action)
        self.start_btn.pack(side="left", padx=5)

        self.save_btn = tk.Button(button_frame, text="保存认定", command=self.save_action)
        self.save_btn.pack(side="left", padx=5)

        self.redraw_btn = tk.Button(
            button_frame, text="重新画框", command=self.redraw_action
        )
        self.redraw_btn.pack(side="left", padx=5)

        self.reset_btn = tk.Button(button_frame, text="重新认定", command=self.reset_action)
        self.reset_btn.pack(side="left", padx=5)

        self.exit_btn = tk.Button(button_frame, text="退出认定", command=self.exit_action)
        self.exit_btn.pack(side="left", padx=5)

        text_frame = tk.Frame(root)
        text_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(text_frame, text="标签:").pack(side="left")
        self.text_entry = tk.Entry(text_frame, width=30)
        self.text_entry.pack(side="left", fill=tk.X, expand=True, padx=5)

        # 绑定鼠标
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 图像显示线程
        self.update_frame()

        rospy.loginfo("PresetAnnotator initialized.")

    def on_close(self):
        self.exit_action()

    # ROS回调
    def rgb_callback(self, msg):
        if not self.freeze:  # 如果冻结，则不更新
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.orig_img = cv_img  # 保存原始图像
                # 计算缩放比例
                h, w = cv_img.shape[:2]
                self.scale_x = w / 640.0
                self.scale_y = h / 480.0
                # 缩放图像用于显示
                self.cv_img = cv2.resize(cv_img, (640, 480))

    # Tkinter更新画布
    def update_frame(self):
        if not self.freeze or self.rect is None:  # 冻结时只更新一次
            with self.lock:
                frame = self.cv_img.copy() if self.cv_img is not None else None

            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(img_rgb)
                self.tk_img = ImageTk.PhotoImage(img_pil)
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # 如果有框，画上去
        if self.rect:
            self.canvas.lift(self.rect)

        self.root.after(30, self.update_frame)

    # 鼠标交互
    def on_press(self, event):
        if self.freeze:  # 只有冻结时能画框
            self.start_x, self.start_y = event.x, event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.start_x,
                self.start_y,
                self.start_x,
                self.start_y,
                outline="red",
                width=2,
            )

    def on_drag(self, event):
        if self.freeze and self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if self.freeze and self.rect:
            # 确保坐标顺序正确
            coords = self.canvas.coords(self.rect)
            x1, y1, x2, y2 = coords
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            self.canvas.coords(self.rect, x1, y1, x2, y2)
            print(f"BBox: {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")

    # 按钮逻辑
    def start_action(self):
        self.freeze = True
        print("开始认定：画面冻结")
        messagebox.showinfo("状态", "画面已冻结，请绘制边界框")

    def redraw_action(self):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        if self.freeze:
            self.update_frame()  # 强制刷新一次画布

    def _encode_image_clip(self, rgb_block: np.ndarray):
        """Extract CLIP features for RGB blocks."""
        clip_request = CLIPRequest()
        clip_request.mode = "encode_image"
        clip_request.image = self.cv_bridge.cv2_to_imgmsg(rgb_block, "rgb8")
        clip_response = self.clip_client(clip_request)
        return clip_response.clip_fts

    def _encode_image_dinov2(self, rgb_block: np.ndarray):
        """Extract DINOv2 features for RGB blocks."""
        dinov2_request = DINOv2Request()
        dinov2_request.mode = "encode_image"
        dinov2_request.image = self.cv_bridge.cv2_to_imgmsg(rgb_block, "rgb8")
        dinov2_response = self.dinov2_client(dinov2_request)
        return dinov2_response.dinov2_fts

    def _encode_image_dinov3(self, rgb_block: np.ndarray):
        """Extract DINOv3 features for RGB blocks."""
        dinov3_request = DINOv3Request()
        dinov3_request.mode = "encode_image"
        dinov3_request.image = self.cv_bridge.cv2_to_imgmsg(rgb_block, "rgb8")
        dinov3_response = self.dinov3_client(dinov3_request)
        return dinov3_response.dinov3_fts

    def _encode_text_clip(self, query_text: str):
        """Encode text using CLIP."""
        req = CLIPRequest()
        req.mode = "encode_text"
        req.text = query_text
        res = self.clip_client(req)
        return res.clip_fts

    def _encode_text_dinov2(self, query_text: str):
        """Encode text using DINOv2."""
        req = DINOv2Request()
        req.mode = "encode_text"
        req.text = query_text
        res = self.dinov2_client(req)
        return res.dinov2_fts

    def _encode_text_dinov3(self, query_text: str):
        """Encode text using DINOv3."""
        req = DINOv3Request()
        req.mode = "encode_text"
        req.text = query_text
        res = self.dinov3_client(req)
        return res.dinov3_fts

    def save_action(self):
        if not self.rect:
            messagebox.showerror("错误", "请先画框")
            return

        text = self.text_entry.get().strip()
        if not text:
            messagebox.showerror("错误", "文本不能为空")
            return

        # 获取画布坐标并转换到原始图像
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        orig_x1 = int(x1 * self.scale_x)
        orig_y1 = int(y1 * self.scale_y)
        orig_x2 = int(x2 * self.scale_x)
        orig_y2 = int(y2 * self.scale_y)

        # 确保坐标在图像范围内
        h, w = self.orig_img.shape[:2]
        orig_x1 = max(0, min(orig_x1, w - 1))
        orig_y1 = max(0, min(orig_y1, h - 1))
        orig_x2 = max(0, min(orig_x2, w - 1))
        orig_y2 = max(0, min(orig_y2, h - 1))

        # 提取原始图像的ROI, 并进行CLIP特征提取
        roi = self.orig_img[orig_y1:orig_y2, orig_x1:orig_x2]
        if self.feature_encode_head == "clip":
            semantic_ft_img = self._encode_image_clip(roi)  # List[float]
            semantic_ft_text = self._encode_text_clip(text)  # List[float]
        elif self.feature_encode_head == "dinov2":
            semantic_ft_img = self._encode_image_dinov2(roi)  # List[float]
            semantic_ft_text = self._encode_text_dinov2(text)  # List[float]
        elif self.feature_encode_head == "dinov3":
            semantic_ft_img = self._encode_image_dinov3(roi)  # List[float]
            semantic_ft_text = self._encode_text_dinov3(text)  # List[float]

        # 保存预设
        self.db._update_entry(text, semantic_ft_text, semantic_ft_img)

        print("预设已保存")

        # 回到流程1：恢复订阅
        self.freeze = False
        self.redraw_action()
        self.text_entry.delete(0, tk.END)  # 清空输入框

    def reset_action(self):
        self.freeze = False
        self.redraw_action()
        self.text_entry.delete(0, tk.END)
        print("重新认定，恢复实时画面")
        messagebox.showinfo("状态", "已恢复实时画面")

    def exit_action(self):
        print("退出认定")
        self.root.quit()
        rospy.signal_shutdown("用户退出")


if __name__ == "__main__":
    root = tk.Tk()
    app = PresetAnnotator(root)
    root.mainloop()
