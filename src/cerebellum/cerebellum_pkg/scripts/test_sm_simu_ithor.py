#!/usr/bin/env python
import json
import rospy
import numpy as np
import tf
import tf2_ros
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion
from tf.transformations import euler_from_quaternion
from std_msgs.msg import String
from coarse_localize.srv import Query as CoarseLocalizeQuery
from fine_grained_search.srv import Query as FineGrainedSearchQuery
from coarse_localize.srv import Show


class SMSimuIthorTester:
    def __init__(self):
        # ROS Params
        rospy.init_node("sm_simu_ithor_tester")
        ## General
        self.trace = []
        self.inv_trace = []
        self.trace_poses = []
        self.trace_idx = -1
        ## Input - Pose
        self.pose_link_father = rospy.get_param("~pose_link_father", "map")
        self.pose_link_child = rospy.get_param("~pose_link_child", "camera_link")
        # Subscriber & Publisher
        ## Subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.user_sub = rospy.Subscriber("~user", String, self.user_callback)
        self.simu_keyboard_sub = rospy.Subscriber(
            "/ithor/keyboard_cmd", String, self.keyboard_callback
        )
        ## Publisher
        self.marker_pub = rospy.Publisher("~nav_marker", Marker, queue_size=10)
        self.simu_teleport_pub = rospy.Publisher(
            "/ithor/teleport_cmd", String, queue_size=10
        )
        self.simu_keyboard_pub = rospy.Publisher(
            "/ithor/keyboard_cmd", String, queue_size=10
        )

        # Server & Client
        rospy.wait_for_service("/memory_manager/memory_show")
        self.coarse_localize_show_client = rospy.ServiceProxy(
            "/memory_manager/memory_show", Show
        )
        rospy.loginfo("Waiting for Query service...")
        rospy.wait_for_service("/memory_manager/memory_query")
        self.coarse_localize_client = rospy.ServiceProxy(
            "/memory_manager/memory_query", CoarseLocalizeQuery
        )
        rospy.wait_for_service("/fine_grained_search_node/fine_query")
        self.fine_grained_search_client = rospy.ServiceProxy(
            "/fine_grained_search_node/fine_query", FineGrainedSearchQuery
        )
        rospy.loginfo("SM Simu Tester initialized.")
        # Timer to show memory markers
        rospy.Timer(rospy.Duration(1), self.memory_makers_show)

    def memory_makers_show(self, event):
        self.coarse_localize_show_client("all")

    def get_current_pose(self):
        """Get the pose when key is pressed."""
        try:
            transform = self.tf_buffer.lookup_transform(  # Lookup TF to get the pose
                self.pose_link_father,
                self.pose_link_child,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
            t = transform.transform.translation
            q = transform.transform.rotation
            return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
        except Exception as e:
            rospy.logwarn(f"TF lookup failed in sync_callback: {e}")

    def keyboard_callback(self, msg):
        """Handle keyboard command callback."""
        command = msg.data
        pairs = {"W": "S", "A": "D", "S": "W", "D": "A"}
        self.trace_idx += 1
        self.trace_poses.append(self.get_current_pose())
        self.trace.append(command)
        self.inv_trace.append(pairs[command])

        rospy.loginfo(
            f"Trace idx {self.trace_idx}: {command}, pose: {self.trace_poses[-1]}"
        )

    def find_best_match_trace(self, target_pose):
        """Find the best matching trace for the given target pose."""
        if not self.trace_poses:
            return None
        distances = [
            np.linalg.norm(np.array(target_pose) - np.array(pose))
            for pose in self.trace_poses
        ]
        min_distance = np.min(distances)
        candidate_indices = [
            i
            for i, d in enumerate(distances)
            if np.isclose(d, min_distance, rtol=1e-5, atol=1e-5)
        ]
        return max(candidate_indices)

    def user_callback(self, msg):
        """Handle user command callback."""
        target = msg.data
        rospy.loginfo(f"Received user command: {target}")
        self.execute_command(target)

    def execute_command(self, target):
        """Execute the user command."""
        # Coarse localization
        try:
            response = self.coarse_localize_client(target)
            candidate_poses = response.poses  # rad
        except rospy.ServiceException as e:
            rospy.logerr(f"Coarse localization service call failed: {e}")

        # if candidate_poses:
        #     best_match_pose = candidate_poses[0]
        #     best_match_pose_list = [
        #         best_match_pose.position.x,
        #         best_match_pose.position.y,
        #         best_match_pose.position.z,
        #         best_match_pose.orientation.x,
        #         best_match_pose.orientation.y,
        #         best_match_pose.orientation.z,
        #         best_match_pose.orientation.w,
        #     ]
        #     best_match_trace_idx = self.find_best_match_trace(best_match_pose_list)
        #     if best_match_trace_idx is not None:
        #         rospy.logwarn(
        #             f"Moving to best match trace index: {best_match_trace_idx}"
        #         )
        #         rospy.sleep(1)
        #         # Move along the trace from current to best_match_trace_idx
        #         steps = self.trace_idx - best_match_trace_idx
        #         if steps <= 0:
        #             inv_cmd_list = []
        #         else:
        #             inv_cmd_list = self.inv_trace[-steps:]
        #             inv_cmd_list = inv_cmd_list[::-1]  # Reverse the list
        #         for i, inv_cmd in enumerate(inv_cmd_list):
        #             self.simu_keyboard_pub.publish(inv_cmd)
        #             rospy.loginfo(f"Going {i+1}/{steps}")
        #             rospy.sleep(1)
        #         rospy.logwarn(f"Arrived at trace index: {self.trace_idx}")
        #     else:
        #         rospy.logwarn("No matching trace index found.")
        # else:
        #     rospy.logwarn("No candidate poses found.")
        #     return

        # rospy.sleep(1)
        # # Fine-grained search
        # try:
        #     response = self.fine_grained_search_client(target)
        #     rospy.loginfo(f"Fine-grained search response: {response}")
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"Fine-grained search service call failed: {e}")


if __name__ == "__main__":
    try:
        node = SMSimuIthorTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
