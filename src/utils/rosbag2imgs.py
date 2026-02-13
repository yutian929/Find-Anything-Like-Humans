#!/usr/bin/env python3
# filepath: /home/yutian/YanBot/scripts/extract_images_from_bag.py

import rosbag
import cv2
from cv_bridge import CvBridge
import os
from tqdm import tqdm
import argparse


def extract_images_from_bag(bag_path, output_dir, image_topic, encoding="bgr8"):
    """
    Extract images from a rosbag file and save them to the specified directory

    Args:
        bag_path (str): Path to the rosbag file
        output_dir (str): Directory to save the extracted images
        image_topic (str): Name of the image topic to extract
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the CvBridge
    bridge = CvBridge()

    # Open the rosbag
    bag = rosbag.Bag(bag_path)

    # Get total number of messages for progress bar
    total_msgs = bag.get_message_count(topic_filters=[image_topic])

    print(f"Extracting images from topic: {image_topic}")

    # Extract images
    for i, (topic, msg, t) in enumerate(
        tqdm(bag.read_messages(topics=[image_topic]), total=total_msgs)
    ):
        try:
            # Convert ROS image message to OpenCV image
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

            # Generate filename using timestamp
            filename = os.path.join(output_dir, f"frame_{t.to_nsec()}.jpg")

            # Save the image
            cv2.imwrite(filename, cv_img)

        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            continue

    bag.close()
    print(f"Extraction complete. Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from rosbag")
    parser.add_argument(
        "--bag_path", default="A328_2D_Mapping.bag", help="Path to the rosbag file"
    )
    parser.add_argument(
        "--output_dir",
        default="default_imgs",
        help="Directory to save extracted images",
    )
    parser.add_argument(
        "--topic",
        default="/camera/image_raw",
        help="Image topic name (default: /camera/image_raw)",
    )
    parser.add_argument(
        "--encoding",
        default="bgr8",
        help="Desired encoding for the image (default: bgr8)",
    )

    args = parser.parse_args()

    extract_images_from_bag(args.bag_path, args.output_dir, args.topic, args.encoding)


if __name__ == "__main__":
    main()
