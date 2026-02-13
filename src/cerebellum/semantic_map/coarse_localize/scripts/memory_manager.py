import time

import tf
import rospy
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray

from coarse_localize.msg import Memory
from coarse_localize.srv import (
    Query,
    QueryResponse,
    Show,
    ShowResponse,
)
from clip_pkg.srv import CLIP, CLIPRequest
from dinov2_pkg.srv import DINOv2, DINOv2Request, DINOv2Response
from dinov3_pkg.srv import DINOv3, DINOv3Request, DINOv3Response
from memory_database import MemoryDB
from preset_database import PresetDB


class MemoryManager:
    def __init__(self):
        rospy.init_node("memory_manager")
        # ROS Parameters
        ## General
        self.debug = rospy.get_param("~debug", True)
        self.feature_encode_head = rospy.get_param(
            "~feature_encode_head", "dinov3"
        )  # clip / dinov2 / dinov3
        ## Input - TF(same as generator)
        self.pose_link_father = rospy.get_param("~pose_link_father", "map")
        ## Processing - Memory DB
        memory_db_path = rospy.get_param("~memory_db_path", "coarse_localize.db")
        renew_db = rospy.get_param("~renew_db", True)
        self.grid_size = rospy.get_param("~grid_size", 0.05)
        self.max_yaws_per_grid = rospy.get_param("~max_yaws_per_grid", 4)
        self.query_top_k = rospy.get_param("~query_top_k", 5)
        self.query_similarity_threshold = rospy.get_param(
            "~query_similarity_threshold", 0.15
        )
        ## Processing - Preset DB
        preset_db_path = rospy.get_param("~preset_db_path", "preset.db")
        self.query_preset_threshold = rospy.get_param("~query_preset_threshold", 0.95)
        ## Output
        self.enable_show_query_markers = rospy.get_param(
            "~enable_show_query_markers", True
        )

        # Subscriber & Publisher
        ## Sub memory
        self.memory_sub = rospy.Subscriber(
            "memory_generator/memory", Memory, self._memory_callback, queue_size=1
        )
        ## Publisher
        self.memory_markers_pub = rospy.Publisher(
            "~memory_markers", MarkerArray, queue_size=1
        )
        self.query_markers_pub = rospy.Publisher(
            "~query_markers", MarkerArray, queue_size=1
        )

        # Initialize database
        self.memory_db = MemoryDB(
            db_path=memory_db_path,
            renew_db=renew_db,
            grid_size=self.grid_size,
            max_yaws_per_grid=self.max_yaws_per_grid,
        )
        self.preset_db = PresetDB(
            db_path=preset_db_path,
            renew_db=False,
        )

        # Server & Client
        ## Show memory markers
        self.show_server = rospy.Service("~memory_show", Show, self._handle_show)
        ## Query memory
        self.query_server = rospy.Service("~memory_query", Query, self._handle_query)
        ## feature_encode
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

        rospy.loginfo("MemoryManager initialized.")

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

    def create_del_all_marker_array(self):
        """Create a marker array to delete all markers in the map."""
        delete_marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self.pose_link_father
        delete_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(delete_marker)
        return delete_marker_array

    def create_marker_by_xyyaw(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        id: int = 0,
        r: float = 1.0,
        g: float = 0.0,
        b: float = 0.0,
        conf: float = 1.0,
    ):
        """Convert grid coordinates and yaw to a Marker for visualization."""
        marker = Marker()
        marker.header.frame_id = self.pose_link_father
        marker.header.stamp = rospy.Time.now()
        marker.ns = "memory_arrows"
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        marker.scale.x = 0.15 * conf
        marker.scale.y = 0.02 * conf
        marker.scale.z = 0.02 * conf
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        return marker

    def create_pose_by_xyyaw(self, x: float, y: float, yaw: float):
        """Create a Pose message from x, y, and yaw."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _memory_callback(self, msg: Memory):
        """Process incoming Memory messages and store them in the database."""
        time_dict = {"total": time.time()}
        try:
            pose_params = msg.pose_params  # Extract pose and clip features
            if len(pose_params) != 7:
                rospy.logerr(f"Invalid pose parameters shape: {len(pose_params)}")
                return
            x, y, z, qx, qy, qz, qw = pose_params
            yaw = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])[2]
            semantic_ft_dim = msg.semantic_ft_dim
            features = [
                list(msg.semantic_fts[i * semantic_ft_dim : (i + 1) * semantic_ft_dim])
                for i in range(len(msg.semantic_fts) // semantic_ft_dim)
            ]
            # Store x-y-yaw-features into the database
            self.memory_db._single_process(x, y, z, yaw, features)
            time_dict["total"] = time.time() - time_dict["total"]
            if self.debug:
                rospy.loginfo(f"{'*' * 10} Memory Manager Debug Info {'*' * 10}")
                for key, val in time_dict.items():
                    rospy.loginfo(f"{key}: {val:.3f} seconds")
        except Exception as e:
            rospy.logerr(f"Error processing memory_callback: {str(e)}")

    def _handle_show(self, req: Show):
        """Handle the Show service request."""
        try:
            del_marker_array = self.create_del_all_marker_array()
            self.memory_markers_pub.publish(del_marker_array)
            if req.data == "all":
                all_world_xyyaw = self.memory_db._get_all_xyyaw()
                if not all_world_xyyaw:
                    rospy.logwarn("No data found in the database.")
                    return ShowResponse(success=False, message="No data found.")
                else:
                    memory_marker_array = MarkerArray()
                    for i, (x, y, yaw) in enumerate(all_world_xyyaw):
                        marker = self.create_marker_by_xyyaw(x, y, 0.0, yaw, i)
                        memory_marker_array.markers.append(marker)
                    self.memory_markers_pub.publish(memory_marker_array)
                    return ShowResponse(
                        success=True,
                        message=f"Published {len(all_world_xyyaw)} markers successfully.",
                    )
            else:
                rospy.logwarn("Invalid request data for Show service")
                return ShowResponse(success=False, message="Invalid request data")
        except Exception as e:
            rospy.logerr(f"Error in handle_show: {str(e)}")
            return ShowResponse(success=False, message=str(e))

    def _handle_query(self, req: Query):
        """Handle the Query service request."""
        try:
            # breakpoint()
            query = req.query
            res = QueryResponse()
            if self.feature_encode_head == "clip":
                query_ft = self._encode_text_clip(query)  # List[float]
            elif self.feature_encode_head == "dinov2":
                query_ft = self._encode_text_dinov2(query)  # List[float]
            elif self.feature_encode_head == "dinov3":
                query_ft = self._encode_text_dinov3(query)  # List[float]
            else:
                rospy.logerr("Unsupportable feature_encode_head !")
                return res

            # query preset db first
            preset_res = self.preset_db.query_by_semantic_text_ft(query_ft, 1)
            if (
                preset_res
                and preset_res[0]["similarity"] >= self.query_preset_threshold
            ):
                query_ft = preset_res[0][
                    "semantic_ft_img"
                ]  # replace query_ft with preset image feature
                rospy.logwarn(
                    f"Preset query result: {preset_res[0]['text']}, {preset_res[0]['similarity']:.3f}. Using preset_img_ft as query_tf."
                )
            # query memory db
            if query_ft is not None:
                query_res = self.memory_db.query_by_ft(query_ft, self.query_top_k, "xy")
                if query_res:
                    best_match = query_res[0]
                    if best_match["similarity"] >= self.query_similarity_threshold:
                        res.success = True
                        res.message = f"Best match Grid: {best_match['x']}, {best_match['y']}, yaw = {best_match['yaw']}, Similarity: {best_match['similarity']:.3f}\nAll query res = {query_res}"
                        res.center = [
                            best_match["x"],
                            best_match["y"],
                            0.0,
                        ]
                        res.poses = [
                            self.create_pose_by_xyyaw(m["x"], m["y"], m["yaw"])
                            for m in query_res
                        ]
                        # show query res markers
                        if self.enable_show_query_markers:
                            del_markers = self.create_del_all_marker_array()
                            self.query_markers_pub.publish(del_markers)
                            query_marker_array = MarkerArray()
                            for i, match in enumerate(query_res):
                                x, y = match["x"], match["y"]
                                yaw = match["yaw"]
                                marker = self.create_marker_by_xyyaw(
                                    x,
                                    y,
                                    0.1,
                                    yaw,
                                    i,
                                    0.0,
                                    1.0,  # green
                                    0.0,
                                    (self.query_top_k - i)
                                    / self.query_top_k,  # seq to decide size
                                )
                                query_marker_array.markers.append(marker)
                            self.query_markers_pub.publish(query_marker_array)
                    else:
                        res.success = False
                        res.message = (
                            f"No suitable match found, query_res = {query_res}"
                        )
                else:
                    res.success = False
                    res.message = "No matches found in the database"
            else:
                res.success = False
                res.message = "Failed to clip query ft"
            return res
        except Exception as e:
            rospy.logerr(f"Error in handle_query: {str(e)}")
            return QueryResponse(success=False, message=str(e))


if __name__ == "__main__":
    try:
        node = MemoryManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error in MemoryManager: {e}")
