#!/bin/bash

# Define the URLs for the checkpoints
BASE_URL="https://github.com/ultralytics/assets/releases/download/"
YOLOv8l_worldv2_url="${BASE_URL}v8.2.0/yolov8l-worldv2.pt"

# Function to download a file if it doesn't already exist
download_if_not_exists() {
    local url=$1
    local filename=$(basename $url)
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename checkpoint..."
        wget $url || { echo "Failed to download checkpoint from $url"; exit 1; }
    else
        echo "$filename already exists, skipping download."
    fi
}

# Download each of the checkpoints
download_if_not_exists $YOLOv8l_worldv2_url

echo "All yolo-world checkpoints are downloaded successfully."
