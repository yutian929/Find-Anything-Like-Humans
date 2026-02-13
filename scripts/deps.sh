#!/bin/bash

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error occurred in the previous command. Exiting."
        exit 1
    fi
}

check_ws() {
    if [ -z "$YANBOT_WS" ]; then
        echo "YANBOT_WS is not set. Exiting."
        exit 1
    else 
        echo "YANBOT_WS is set to $YANBOT_WS"
        cd $YANBOT_WS
    fi
}

check_folder() {
    local folder=$1
    if [ -d "$folder" ]; then
        return 0
    else
        return 1
    fi
}



# --------------------------------------------------------------------------------------------------------------------------------
# ROS deps
check_ws
sudo apt install -y udev can-utils libeigen3-dev libpcap-dev python3-testresources libcjson1 libcjson-dev libpcl-dev v4l-utils
check_success
sudo apt install ros-noetic-gmapping ros-noetic-rtabmap-ros ros-noetic-joy ros-noetic-robot-pose-ekf ros-noetic-image-transport ros-noetic-rgbd-launch ros-noetic-ddynamic-reconfigure ros-noetic-octomap-server 
check_success
rosdep install --from-paths src --ignore-src -r -y
check_success
# --------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------
# camera
## Realsense
### librealsense
check_ws
sudo apt install libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev 
mkdir -p thirdparties/
cd thirdparties/

if check_folder "librealsense"; then
    echo "Folder 'librealsense' already exists, skipping clone and build."
else
    echo "Cloning librealsense..."
    git clone https://github.com/yutian929/YanBot-librealsense.git librealsense
    check_success
    cd librealsense
    ./scripts/setup_udev_rules.sh
    check_success
    if check_folder "build"; then
        echo "Removing existing librealsense build directory..."
        rm -rf build
    fi
    mkdir build && cd build
    cmake ..
    check_success
    make -j$(( $(nproc) / 2 ))
    check_success
    sudo make install
    check_success
fi
### realsense-ros
# check_ws
# mkdir -p thirdparties/realsense_ros/src/
# cd thirdparties/realsense_ros/src/
# if check_folder "realsense_ros"; then
#     echo "Folder 'realsense_ros' already exists, skipping clone."
# else
#     echo "Cloning realsense_ros..."
#     git clone https://github.com/yutian929/YanBot-realsense_ros.git realsense_ros
#     check_success
#     source /opt/ros/noetic/setup.sh
#     cd ..
#     catkin_make
#     check_success
#     echo "source $YANBOT_WS/thirdparties/realsense_ros/devel/setup.bash" >> ~/.bashrc
# fi
# --------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------
# Arm
## ARX-R5
sudo apt install udev can-utils
check_success
# --------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------
# Brain
pip install sxtwl requests geopy httpx[socks] qrcode[pil]
pip install -U openai
check_success
# --------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------
# SLAM
## MID360
### Livox-SDK2
check_ws
mkdir -p thirdparties/
cd thirdparties/
if check_folder "Livox-SDK2"; then
    echo "Folder 'Livox-SDK2' already exists, skipping clone and build."
else
    git clone "https://github.com/Livox-SDK/Livox-SDK2.git" "Livox-SDK2"
    check_success
    cd Livox-SDK2
    mkdir -p build
    cd build
    cmake .. && make -j
    check_success
    sudo make install
    check_success
fi
### Livox ROS Driver 2
check_ws
mkdir -p thirdparties/Livox-ROS-Driver2/src/
cd thirdparties/Livox-ROS-Driver2/src/
if check_folder "Livox-ROS-Driver2"; then
    echo "Folder 'Livox-ROS-Driver2' already exists, skipping clone and build."
else
    git clone "https://github.com/Livox-SDK/livox_ros_driver2.git" "Livox-ROS-Driver2"
    check_success
    cd Livox-ROS-Driver2
    source /opt/ros/noetic/setup.sh
    ./build.sh ROS1
    echo "source $YANBOT_WS/thirdparties/Livox-ROS-Driver2/devel/setup.bash" >> ~/.bashrc
fi
## FAST_LIO && LOCALIZATION
check_ws
sudo apt install ros-noetic-ros-numpy ros-noetic-octomap-ros ros-noetic-octomap-server ros-noetic-pointcloud-to-laserscan
pip install numpy==1.21
pip install open3d
### repo
# --------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------
# interact
sudo apt install swig libatlas-base-dev libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 python3-pyaudio sox
check_ws
echo "====================================================="
echo "YanBot requires a conda environment named 'interact'"
echo "====================================================="
echo ""
echo "Please complete the following steps:"
echo "1. Install Miniconda."
echo ""
echo "2. Create/update the interact environment:"
echo "     conda env create -f scripts/interact.yaml -v"
echo "     conda env update -n interact -f scripts/py310.yaml -v"
echo ""
echo "====================================================="
read -p "Have you completed the above steps? (y/n): " answer
if [[ "$answer" != "y" ]]; then
    echo "Please complete the steps and run this script again."
    exit 1
fi

# --------------------------------------------------------------------------------------------------------------------------------
# semantic_map
pip install git+https://github.com/openai/CLIP.git
pip install networkx==3.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
## yoloesam
check_ws
echo "====================================================="
echo "YanBot requires a conda environment named 'py310'"
echo "====================================================="
echo ""
echo "Please complete the following steps:"
echo "1. Install Miniconda."
echo ""
echo "2. Create/update the py310 environment:"
echo "     conda env create -f scripts/py310.yaml -v"
echo "     conda env update -n py310 -f scripts/py310.yaml -v"
echo ""
echo "====================================================="
read -p "Have you completed the above steps? (y/n): " answer
if [[ "$answer" != "y" ]]; then
    echo "Please complete the steps and run this script again."
    exit 1
fi

cd src/Cerebellum/semantic_map/yolo_evsam_ros/weights/
./download_weights.sh
# --------------------------------------------------------------------------------------------------------------------------------
