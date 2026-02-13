#!/usr/bin/env python
import rospy
import json
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion
from tf.transformations import euler_from_quaternion
from std_msgs.msg import String
from asm.srv import Guide, GuideRequest, Query, QueryRequest


class ASMTester:
    def __init__(self):
        rospy.init_node("asm_tester")

        # 等待服务初始化
        rospy.loginfo("Waiting for Query and Guide service...")
        rospy.wait_for_service("/asm_manager/query")
        self.query_client = rospy.ServiceProxy("/asm_manager/query", Query)
        rospy.wait_for_service("/asm_manager/guide")
        self.guide_client = rospy.ServiceProxy("/asm_manager/guide", Guide)

        # 初始化 move_base action client
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()

        # 初始化导航箭头可视化发布器
        self.marker_pub = rospy.Publisher("~nav_marker", Marker, queue_size=10)

        # 添加目标指令订阅者
        self.target_sub = rospy.Subscriber("~target", String, self.target_callback)

        rospy.loginfo("ASM Tester initialized complete.")

    def send_navigation_goal(self, pose):
        """发送单个导航目标到 move_base，并等待结果"""
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = pose
        # 发送目标并指定反馈回调
        self.move_base_client.send_goal(goal, feedback_cb=self.navigation_feedback)
        rospy.loginfo("导航目标已发送，等待执行结果...")
        # 设置超时（30秒）
        finished = self.move_base_client.wait_for_result(rospy.Duration(30))
        if not finished:
            rospy.logerr(
                f"导航失败: 超时 - 目标({pose.position.x:.2f}, {pose.position.y:.2f})未在30秒内到达"
            )
            self.move_base_client.cancel_goal()
            return False

        state = self.move_base_client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("成功到达目标位置！")
            return True
        else:
            # 获取状态文本描述
            status_dict = {
                actionlib.GoalStatus.PENDING: "等待执行",
                actionlib.GoalStatus.ACTIVE: "正在执行",
                actionlib.GoalStatus.PREEMPTED: "导航被取消",
                actionlib.GoalStatus.ABORTED: "导航被终止(可能遇到障碍物或无法规划路径)",
                actionlib.GoalStatus.REJECTED: "导航目标被拒绝",
                actionlib.GoalStatus.LOST: "导航目标丢失",
            }
            status_text = status_dict.get(state, f"未知状态({state})")
            error_msg = self.move_base_client.get_goal_status_text() or "无详细信息"
            rospy.logerr(f"导航失败: {status_text} - {error_msg}")
            rospy.logwarn(f"目标位置: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
            return False

    def navigation_feedback(self, feedback):
        """处理导航反馈"""
        current_pose = feedback.base_position.pose
        yaw = euler_from_quaternion(
            [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
        )[2]
        rospy.loginfo_throttle(
            1.0,
            "实时位置: ({:.2f}, {:.2f}), 朝向: {:.2f} rad".format(
                current_pose.position.x, current_pose.position.y, yaw
            ),
        )

    def publish_marker(self, nav_pose, idx):
        """发布箭头 Marker 用于 RViz 可视化"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nav_arrow"
        marker.id = idx
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start_point = nav_pose.position
        quat = nav_pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        end_point = Point()
        end_point.x = start_point.x + math.cos(yaw)
        end_point.y = start_point.y + math.sin(yaw)
        end_point.z = start_point.z

        marker.points = [start_point, end_point]
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def go_to_target(self, target):
        """导航到指定目标"""
        # 调用Query服务获取导航点
        query_req = QueryRequest()
        query_req.query = target
        query_res = self.query_client(query_req)
        if not query_res.success:
            rospy.logerr(f"查询目标失败: {query_res.message}")
            return
        target_id = query_res.id
        target_labels = query_res.labels
        target_center = query_res.center
        rospy.loginfo(
            f"查询结果: ID={target_id}, Labels={target_labels}, Center={target_center}"
        )
        # 调用Guide服务获取导航点
        guide_req = GuideRequest()
        guide_req.guide_type = "near"
        goal_pose = Pose()
        goal_pose.position.x = target_center[0]
        goal_pose.position.y = target_center[1]
        goal_pose.position.z = target_center[2]
        goal_pose.orientation = Quaternion(0, 0, 0, 1)
        guide_req.goal_pose = goal_pose
        guide_res = self.guide_client(guide_req)
        if not guide_res.success:
            rospy.logerr(f"导航目标获取失败: {guide_res.message}")
            return
        # 依次尝试导航到每个可能的目标点
        for idx, nav_pose in enumerate(guide_res.navi_goals):
            rospy.loginfo(f"尝试导航点 {idx+1}/{len(guide_res.navi_goals)} : {nav_pose}")
            self.publish_marker(nav_pose, idx)
            if self.send_navigation_goal(nav_pose):
                rospy.loginfo(f"成功到达目标位置: {target}")
                return
            else:
                rospy.logwarn(f"导航点 {idx+1} 失败，尝试下一个")

    def target_callback(self, msg):
        """处理目标指令的回调函数"""
        target = msg.data
        rospy.loginfo(f"收到目标指令: {target}")
        self.go_to_target(target)


if __name__ == "__main__":
    try:
        node = ASMTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
