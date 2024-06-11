<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

This repository is dedicated solely to inferencing the CenterPoint-pointpillar model.


## Environment
### 1. Docker Environment
- Base Image: [`nvcr.io/nvidia/tensorrt:23.04-py3`](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_23-04)
- OS: Ubuntu 20.24
- CUDA: 12.1.0
- cuDNN: 8.9.0
- TensorRT: 8.6.1
- python: 3.8
- Pytorch: 2.1.1
### 2. Docker Container Start
Please refer to the [docker/README.md](docker/README.md) for more details.
### 3. PCDET Installation
``` shell
cd ~/OpenPCDet
sudo python setup.py develop
```
## Usage
Support the below iference methods:
- ROS2 Python Node
- ROS2 C++ Node (Comming soon....)
### 1. ROS2 Python Node
``` shell
cd tools/
python ros2_demo.py --cfg_file {cfg_path} --ckpt {ckpt_path}
```
<!-- show picture sources/fig1.png-->
<img src="sources/fig1.png" align="center" width="100%">

### 2. ROS2 C++ Node
Build the ROS2 package in your ROS2 workspace.
``` shell
cd ~/ && mkdir -p ros2_ws/src && cd ros2_ws/ && colcon build --symlink-install
cd src && ln -s OPENPCDET_PATH/centerpoint .
cd ../ && colcon build --symlink-install
```
Run the ROS2 Node.
``` shell
ros2 launch centerpoint centerpoint.launch.py
```
## Evaluation
To evaluate TensorRT results, you have to wrap the c++ to python API.
### 1. Build Python module
``` shell
cd centerpoint/pybind
cmake -BRelease
cmake --build Release
```
### 2. Copy Python module
``` shell
cp centerpoint/pybind/tools/cp.cpython-38-x86_64-linux-gnu.so tools/
```
### 3. Evaluation
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