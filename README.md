<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

This repository is dedicated solely to inferencing the CenterPoint-pointpillar model.

# Docker Environment
- Base Image: [`nvcr.io/nvidia/tensorrt:23.04-py3`](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_23-04)
- OS: Ubuntu 20.24
- CUDA: 12.1.0
- cuDNN: 8.9.0
- TensorRT: 8.6.1
- python: 3.8
- Pytorch: 2.1.1


## 1) Set Environment

### 1.2 Install Docker Engine on Ubuntu
Please refer to the [`docker.docs`](https://docs.docker.com/engine/install/ubuntu/) for more details.

### 1.3 Clone this repository

### 1.4 Docker Container Start

#### 1.4.1 Build the docker base image
```shell script
docker build -f docker/env.Dockerfile -t openpcdet-env docker/
```

#### 1.4.2 Start the container.
``` shell
docker compose up --build -d
```

Please refer to the [docker/README.md](docker/README.md) for more details.

### 1.5 PCDET Installation

#### 1.5.1 Execute the container
```
docker exec -it centerpoint bash
```

#### 1.5.2 Install OpenPCDet
``` shell
cd ~/OpenPCDet
sudo python setup.py develop
```

## 2) Usage: Inference Method using ROS2 *Python* Node on the Container ENV

### 2.1.1 ROS2 play bagfile on the container
```
docker exec -it centerpoint bash
cd /Dataset
ros2 bag play segment-10359308928573410754_720_000_740_000_with_camera_labels/  # ros2 bag play folder_with_ros2bag
```

### 2.1.2 execute ros2_demo.py on the container
``` shell
docker exec -it centerpoint bash
cd ~/OpenPCDet/tools/
python ros2_demo.py --cfg_file cfgs/waymo_models/centerpoint_pillar_inference.yaml --ckpt ../ckpt/checkpoint_epoch_24.pth
```

### 2.1.3 execute rviz2
``` shell
docker exec -it centerpoint bash
rviz2
```

### 2.1.4 setting rviz2
- Fixed Frame: base_link
- Add -> By display type -> PountCloud2 -> Topic: /lidar/top/pointcloud, Size(m): 0.03
- Add -> By topic -> /boxes/MarkerArray

<img src="sources/rviz2_add_topic.png" align="center" width="359">

### 2.1.5 run rviz2

<!-- show picture sources/fig1.png-->
<img src="sources/rviz2.png" align="center" width="100%">
<img src="sources/fig1.png" align="center" width="100%">

## 3) Usage: Inference Method using ROS2 *C++* Node on the Container ENV (Comming soon....)

### 3.1 ROS2 C++ Node
Build the ROS2 package in your ROS2 workspace.
``` shell
cd ~/ && mkdir -p ros2_ws/src && cd ros2_ws/ && colcon build --symlink-install
cd src && ln -s OPENPCDET_PATH/centerpoint .
cd ../ && colcon build --symlink-install
```
### 3.2 Run the ROS2 Node.
``` shell
ros2 launch centerpoint centerpoint.launch.py
```

## 4) Evaluation
To evaluate TensorRT results, you have to wrap the c++ to python API.
### 4.1 Build Python module
``` shell
cd centerpoint/pybind
cmake -BRelease
cmake --build Release
```
### 4.2 Copy Python module
``` shell
cp centerpoint/pybind/tools/cp.cpython-38-x86_64-linux-gnu.so tools/
```
### 4.3 Evaluation
``` shell
python test.py --cfg_file cfgs/waymo_models/centerpoint_pillar_inference.yaml --TensorRT
```
```
2024-06-11 05:36:52,125   INFO
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP: 0.5793
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH: 0.5733
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APL: 0.5793
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP: 0.5150
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH: 0.5095
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APL: 0.5150
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP: 0.6583
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH: 0.3265
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APL: 0.6583
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP: 0.5964
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH: 0.2959
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APL: 0.5964
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APL: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APL: 0.0000
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP: 0.4227
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH: 0.3211
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APL: 0.4227
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP: 0.3966
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH: 0.3013
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APL: 0.3966
```
