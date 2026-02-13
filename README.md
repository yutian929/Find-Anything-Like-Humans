# Find Anything Like Humans (FALH)

**Official Code for "Find Anything Like Humans: Online Semantic Mapping And Coarse-to-fine Navigation in Dynamic Environments"**

*Accepted at ICRA 2026*

[![Bilibili](https://img.shields.io/badge/Demo_Video-Bilibili-00A1D6?logo=bilibili&logoColor=white)](https://space.bilibili.com/504386197/lists/7415630?type=season)

---

## Overview

FALH is an online framework for language-guided (or image-guided) object navigation inspired by how humans search: **recalling likely regions from memory, then verifying up close**. The system requires no predefined vocabulary prompts during mapping and works in dynamic environments.

**Core pipeline:**

```
Pre-explore ──► Coarse Localization ──► Fine Search ──► 3D Goal
  (build          (retrieve top-K        (verify target     (depth → point
  scene            poses from memory      in local view      cloud → AABB)
  memory)          via feature sim.)      via seg + matching)
```

**Key contributions:**
- **Human-like coarse-to-fine** search strategy — recall likely regions, then verify locally.
- **Prompt-free perception** — class-agnostic detection & segmentation, no predefined word lists needed.
- **Robot-centric memory** — compact grid-indexed scene memory (<40 MB per 20 m²), auto-updated on revisit.
- Outperforms ConceptGraphs and HOV-SG on HM3D, AI2-THOR, and real-world deployments.

---

## System Architecture

```
┌──────────────── Semantic Map Building (Online) ────────────────┐
│                                                                │
│  RGB Frame ──► FastSAM (full-image seg)                        │
│            ──► MViTs/RAM++ (class-agnostic detection)          │
│            ──► EViT-SAM (bbox-guided seg)                      │
│            ──► Mask Merge (IoU + containment)                  │
│            ──► Feature Encode (CLIP / DINOv2 / DINOv3)         │
│            ──► Grid Memory DB (pose + features → SQLite)       │
│                                                                │
├──────────────── Coarse Localization (Query) ───────────────────┤
│                                                                │
│  Text / Image Query ──► Encode ──► Cosine Sim Search           │
│                     ──► Top-K candidate poses                  │
│                     ──► Visit order (quality + distance)        │
│                                                                │
├──────────────── Fine-Grained Search (at Candidate) ────────────┤
│                                                                │
│  Current View ──► YOLO-World detection (if confident, done)    │
│              ──► Else: full seg pipeline + feature matching     │
│              ──► Best mask → depth → 3D point cloud             │
│              ──► DBSCAN filtering → AABB → 3D goal             │
└────────────────────────────────────────────────────────────────┘
```

The entire system is implemented as **ROS Noetic** service nodes. Heavy models (RAM++, MViTs, CLIP) are isolated in socket workers for GPU efficiency.

---

## Prerequisites

| Component | Version |
|---|---|
| **OS** | Ubuntu 20.04 |
| **ROS** | Noetic |
| **CUDA** | 11.x+ |
| **GPU** | NVIDIA GPU ≥ 8 GB VRAM (tested on RTX 2080 Ti) |
| **Conda** | Miniconda / Anaconda |
| **RAM** | ≥ 16 GB |

---

## Installation

### 1. Clone & Set Up Workspace

```bash
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
git clone https://github.com/yutian929/Find-Anything-Like-Humans.git
cd Find-Anything-Like-Humans

# Run basic.sh to set environment variables (YANBOT_WS, HF mirror, etc.)
bash scripts/basic.sh
source ~/.bashrc
```

### 2. Install System Dependencies

The project provides a one-click script for system-level dependencies (ROS packages, RealSense SDK, Livox SDK, FAST-LIO, etc.):

```bash
cd $YANBOT_WS
bash scripts/deps.sh
```

This script will:
- Install ROS Noetic packages (`gmapping`, `rtabmap-ros`, `octomap-server`, etc.)
- Build **librealsense** from source (for Intel RealSense D435i)
- Build **Livox-SDK2** and **Livox ROS Driver 2** (for MID360 LiDAR)
- Install SLAM dependencies (`ros-noetic-ros-numpy`, `open3d`, etc.)
- Install interaction-related system libraries (`portaudio`, `swig`, etc.)
- Install CLIP and core Python packages (`torch`, `torchvision`, `networkx`, etc.)
- Prompt you to create the conda environments (see below)

### 3. Create Conda Environments

The project uses **3 conda environments**. Pre-built YAML configs are provided in `scripts/`:

#### (a) `py310` — Main environment (most model services)

```bash
# Create from the pinned YAML (recommended — ensures exact reproducibility)
conda env create -f scripts/py310.yaml -v

# Or update an existing environment
conda env update -n py310 -f scripts/py310.yaml -v
```

<details>
<summary><b>Key packages included in py310.yaml</b></summary>

| Package | Version | Purpose |
|---|---|---|
| `python` | 3.10.0 | — |
| `torch` | 2.6.0 | Deep learning backbone |
| `torchvision` | 0.21.0 | Image transforms & models |
| `openai-clip` | 1.0.1 | CLIP text/image encoding |
| `ultralytics` | 8.3.94 | YOLO-World + FastSAM |
| `supervision` | 0.25.1 | Detection annotation tools |
| `open3d` | 0.19.0 | Point cloud processing |
| `scikit-learn` | 1.7.0 | DBSCAN clustering |
| `opencv-python` | 4.11.0 | Image processing |
| `timm` | 1.0.15 | Vision model hub |
| `xformers` | 0.0.29 | Efficient attention |
| `segment-anything` | 1.0 | SAM base |
| `rospkg` | 1.6.0 | ROS Python bindings |
| `ftfy`, `regex`, `omegaconf` | — | CLIP / config utilities |
| `numpy` | 1.24.4 | — |

</details>

#### (b) `mvits` — Class-agnostic detection (MDef-DETR)

```bash
conda create -n mvits python=3.8 -y
conda activate mvits

pip install torch torchvision
pip install transformers==4.5.1   # pinned for MDef-DETR compatibility
pip install numpy pillow tqdm setuptools
```

> **Note:** MDef-DETR requires `transformers~=4.5.1` — do not upgrade.

#### (c) `interact` — Voice interaction (optional, for real robot deployment)

```bash
# Create from the pinned YAML
conda env create -f scripts/interact.yaml -v
```

<details>
<summary><b>Key packages included in interact.yaml</b></summary>

| Package | Version | Purpose |
|---|---|---|
| `python` | 3.10.16 | — |
| `pytorch` | 2.5.0 (CUDA 11.8) | — |
| `torchaudio` | 2.5.0 | Audio processing |
| `funasr` | 1.2.6 | SenseVoice ASR / STT |
| `sounddevice` | 0.5.1 | Audio recording |
| `chattts` | 0.0.0 | TTS synthesis |
| `librosa` | 0.11.0 | Audio features |
| `modelscope` | 1.24.0 | Model hub (SenseVoice) |
| `transformers` | 4.51.3 | Tokenizers |

</details>

### 4. Download Model Weights

Place model weights in the `thirdparties/` directory (git-ignored):

| Model | Weight File | Source |
|---|---|---|
| **CLIP** | Auto-downloaded | `ViT-B/16` via `clip.load()` |
| **EfficientViT-SAM** | `efficientvit_sam_l1.pt` | [EfficientViT](https://github.com/mit-han-lab/efficientvit) |
| **FastSAM** | `FastSAM-x.pt` | [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) |
| **MDef-DETR** | `MDef_DETR_minus_language_r101_epoch10.pth` | [MViTs](https://github.com/seermer/MViTs) |
| **RAM++** | `ram_plus_swin_large_14m.pth` | [RAM](https://github.com/xinyu1205/recognize-anything) |
| **YOLO-World** | `yolov8l-worldv2.pt` | [Ultralytics](https://docs.ultralytics.com/models/yolo-world/) |
| **DINOv2** | Auto-loaded via hub | [DINOv2](https://github.com/facebookresearch/dinov2) |
| **DINOv3** | Auto-loaded via hub | DINOv3 (ViT-L/16 + text encoder) |

> **Tip:** If downloading from HuggingFace is slow, `basic.sh` has already set `HF_ENDPOINT=https://hf-mirror.com` for you.

### 5. Build the Catkin Workspace

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# (Optional) Add to bashrc for convenience
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Quick Start

### Simulation (AI2-THOR)

**Terminal 1** — Launch AI2-THOR scene bridge:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch ithor_pkg ai2thor.launch
```

**Terminal 2** — Launch the full semantic mapping pipeline:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch cerebellum_pkg sm_simu.launch simu_type:=ithor
```

This starts all model services (SAM, MViTs, RAM++, CLIP, DINOv2, DINOv3, YOLO-World), the `memory_generator`, `memory_manager`, `fine_grained_search`, and RViz.

**Terminal 3** — Teleoperate & query:

```bash
source ~/catkin_ws/devel/setup.bash
rosrun cerebellum_pkg test_sm_simu_ithor.py
```

Use keyboard (WASD) to explore the scene. Once memory is built, type a natural-language query (e.g., `"red apple"`) to trigger coarse localization + fine search.

### Simulation (Habitat / HM3D)

```bash
roslaunch cerebellum_pkg sm_simu.launch simu_type:=habitat
rosrun cerebellum_pkg test_sm_simu_habitat.py
```

### Real Robot Deployment

**Terminal 1** — Launch SLAM + sensors:

```bash
roslaunch cerebellum_pkg main.launch slam_type:=fast_lio
```

This launches FAST-LIO2 (with MID360 LiDAR), RealSense D435i, and the full semantic mapping pipeline.

**Terminal 2** — Run the mapping and query nodes:

```bash
rosrun cerebellum_pkg test_asm_node.py
```

---

## ROS Service API

### Coarse Localization — Query

```
rosservice call /coarse_localize/query "query: 'water bottle'"
```

**Request:** `string query`  
**Response:** `poses[]` (top-K candidate poses), `center`, `aabb_min/max`, etc.

### Fine-Grained Search — Query

```
rosservice call /fine_grained_search/query "query: 'water bottle'"
```

**Request:** `string query`  
**Response:** `center` (3D goal position), `aabb_min/max` (bounding box), and publishes target point cloud.

### Memory Visualization

```
rosservice call /coarse_localize/show "data: 'all'"
```

Publishes color-coded RViz markers representing the grid-indexed scene memory.

---

## Key Parameters

Parameters are configured in the launch files (`coarse_localize.launch`, `fine_grained_search.launch`):

| Parameter | Default | Description |
|---|---|---|
| `grid_size` | `0.5` | Memory grid cell size (meters) |
| `yaw_num` | `4` | Number of directional bins per cell |
| `caod_head` | `mvits` | Detector: `mvits` or `ram` (RAM++ → YOLO) |
| `seg_head` | `evit-sam` | Segmenter: `evit-sam` or `fast-sam` |
| `feature_encode_head` | `dinov3` | Encoder: `clip`, `dinov2`, or `dinov3` |
| `top_k` | `5` | Number of candidates for coarse localization |
| `fine_score_threshold` | `0.25` | YOLO confidence for fine search bypass |
| `renew_db` | `true` | Clear memory DB on startup |

---

## Project Structure

```
Find-Anything-Like-Humans/
├── src/
│   ├── cerebellum/                    # Main system
│   │   ├── cerebellum_pkg/            # Top-level launch & test scripts
│   │   │   ├── launch/
│   │   │   │   ├── main.launch        # Real robot deployment
│   │   │   │   ├── sm_simu.launch     # Simulation (iTHOR / Habitat)
│   │   │   │   └── include/           # Per-model launch files
│   │   │   ├── scripts/               # Test nodes
│   │   │   └── rviz/                  # RViz configs
│   │   ├── semantic_map/
│   │   │   ├── coarse_localize/       # Memory building & coarse retrieval
│   │   │   ├── fine_grained_search/   # Local verification & 3D goal
│   │   │   ├── clip_pkg/              # CLIP encoder service
│   │   │   ├── dinov2_pkg/            # DINOv2 encoder service
│   │   │   ├── dinov3_pkg/            # DINOv3 encoder service
│   │   │   ├── evit_sam_pkg/          # EfficientViT-SAM + FastSAM
│   │   │   ├── caod_pkg/              # MViTs (class-agnostic detector)
│   │   │   ├── ram_pkg/               # RAM++ (tagging)
│   │   │   ├── yolo_world_pkg/        # YOLO-World (open-vocab det.)
│   │   │   ├── alpha_clip_pkg/        # Alpha-CLIP (optional)
│   │   │   └── promptda_pkg/          # Depth enhancement (optional)
│   │   └── wheel/                     # Robot platform drivers
│   └── simu/
│       └── ithor_pkg/                 # AI2-THOR / RoboTHOR ROS bridge
└── .gitignore
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026falh,
  title     = {Find Anything Like Humans: Online Semantic Mapping And Coarse-to-fine Navigation in Dynamic Environments},
  author    = {Zhang, Yutian and Zhang, Jianyu and Liu, Mengyuan},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

---

## License

This project is open-sourced for academic research. Please refer to individual submodule licenses for third-party dependencies.

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [DINOv2 / DINOv3](https://github.com/facebookresearch/dinov2)
- [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- [Ultralytics (YOLO-World)](https://github.com/ultralytics/ultralytics)
- [RAM++ (Recognize Anything)](https://github.com/xinyu1205/recognize-anything)
- [MViTs](https://github.com/seermer/MViTs)
- [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)
- [AI2-THOR](https://ai2thor.allenai.org/)
- [Habitat-Sim](https://aihabitat.org/)
