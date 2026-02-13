#!/usr/bin/env python3

import rospy
import signal
import subprocess
import os
import sys


def signal_handler(sig, frame):
    rospy.loginfo("Received shutdown signal, saving map...")

    root_path = os.environ.get("YANBOT_WS", "/home/yutian/YanBot")
    map_path = os.path.join(
        root_path,
        "src/cerebellum/wheel/wheeltec_noetic_src/turn_on_wheeltec_robot/map/WHEELTEC",
    )

    try:
        # 运行map_saver命令保存地图
        subprocess.call(["rosrun", "map_server", "map_saver", "-f", map_path])
        rospy.loginfo(f"Map saved successfully to {map_path}")
    except Exception as e:
        rospy.logerr(f"Failed to save map: {e}")

    # 完成后退出
    rospy.signal_shutdown("Map saved")
    sys.exit(0)


if __name__ == "__main__":
    rospy.init_node("first_mapping_saver_node")

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.loginfo(
        "Auto map saver node started. Map will be saved on shutdown (Ctrl+C)."
    )

    # 保持节点运行直到收到终止信号
    rospy.spin()
