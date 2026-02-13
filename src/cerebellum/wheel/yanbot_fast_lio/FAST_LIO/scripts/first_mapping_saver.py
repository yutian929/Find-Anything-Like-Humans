#!/usr/bin/env python3

import rospy
import subprocess
import os
import re
import time
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


class MapSaver:
    def __init__(self):
        rospy.init_node("map_saver_node")

        # 获取保存路径参数
        self.save_map_topic = rospy.get_param("~save_map_topic", "/map")
        self.full_path = rospy.get_param(
            "~full_save_path",
            "/home/yutian/projs/YanBot/src/cerebellum/slam/FAST_LIO/PCD/scans",
        )

        # 获取保存间隔参数（秒）
        self.save_interval = rospy.get_param("~save_interval", 10.0)  # 默认10秒

        # 记录上次保存时间
        self.last_save_time = 0

        # 获取保存路径的目录部分
        self.save_dir = os.path.dirname(self.full_path)

        # 创建保存路径（如果不存在）
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 创建地图订阅者
        self.map_sub = rospy.Subscriber(
            self.save_map_topic, OccupancyGrid, self.map_callback
        )

        rospy.loginfo(
            "Map saver node initialized. Will save to: %s every %.1f seconds"
            % (self.full_path, self.save_interval)
        )

    def map_callback(self, map_msg):
        # 检查是否到了保存间隔
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return

        rospy.loginfo("Saving latest map...")

        # 使用map_saver保存地图
        try:
            # 使用subprocess调用map_saver
            cmd = [
                "rosrun",
                "map_server",
                "map_saver",
                "-f",
                self.full_path,
                f"map:={self.save_map_topic}",
            ]
            subprocess.run(cmd, check=True)
            rospy.logwarn(
                "Map saved successfully to %s.pgm and %s.yaml"
                % (self.full_path, self.full_path)
            )

            # 修复yaml文件中的nan值
            self.fix_yaml_file()

            # 更新上次保存时间
            self.last_save_time = current_time

            # 通知用户
            rospy.loginfo("Map processing complete!")

        except subprocess.CalledProcessError as e:
            rospy.logerr("Failed to save map: %s" % str(e))

    def fix_yaml_file(self):
        """处理YAML文件中的nan值"""
        yaml_file = self.full_path + ".yaml"

        if not os.path.exists(yaml_file):
            rospy.logerr("YAML file not found: %s" % yaml_file)
            return

        rospy.loginfo("Checking YAML file for NaN values: %s" % yaml_file)

        # 读取YAML文件
        with open(yaml_file, "r") as f:
            content = f.read()

        # 替换nan为0.0（大小写不敏感）
        modified_content = re.sub(r"\bnan\b", "0.0", content, flags=re.IGNORECASE)

        # 写回文件
        with open(yaml_file, "w") as f:
            f.write(modified_content)

        rospy.loginfo("YAML file processed: NaN values replaced with 0.0")


if __name__ == "__main__":
    try:
        map_saver = MapSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
