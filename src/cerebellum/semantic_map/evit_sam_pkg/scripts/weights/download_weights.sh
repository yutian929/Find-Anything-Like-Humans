#!/bin/bash

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

# Define the URLs for the checkpoints
BASE_URL="https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/"
efficientvit_sam_l2_url="${BASE_URL}efficientvit_sam_l2.pt"
efficientvit_sam_l1_url="${BASE_URL}efficientvit_sam_l1.pt"

download_if_not_exists $efficientvit_sam_l1_url

echo "All EfficientViT-SAM checkpoints are downloaded successfully."
