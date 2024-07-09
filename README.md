
# PointPillar based CenterPoint

## OpenPCDet
<img src="docs/open_mmlab.png" align="right" width="30%">
- `OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 
- This repository is dedicated solely to inferencing the CenterPoint-pointpillar model.

## Docker Environment
- Base Image: [`nvcr.io/nvidia/tensorrt:23.04-py3`](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_23-04)
- OS: Ubuntu 20.24
- CUDA: 12.1.0
- cuDNN: 8.9.0
- TensorRT: 8.6.1
- python: 3.8
- Pytorch: 2.1.1

## 1. Setting Dev Environment using Docker Env 
- Please follow [docs/1_Setting_Environment_using_Docker.md](docs/1_Setting_Environment_using_Docker.md) and proceed with the instructions.

## 2 Preparing Datasets 
- Please follow [docs/2_Preparing_Waymo_Dataset.md](docs/2_Preparing_Waymo_Dataset.md) and proceed with the instructions.

## 3. Setting OpenPCDet.
- Please follow [docs/3_Setting_OpenPCDet.md](docs/3_Setting_OpenPCDet.md) and proceed with the instructions.

## 4. Training a model.
- Please follow [docs/4_Training.md](docs/4_Training.md) and proceed with the instructions.

## 5. Testing a model and Evaluation with Pytorch models.
- Please follow [docs/5_Testing_Evaluation.md](docs/5_Testing_Evaluation.md) and proceed with the instructions.

## 6. Convert ONNX models from Pytorch models
- Please follow [docs/6_Convert_Models.md](docs/6_Convert_Models.md) and proceed with the instructions.

## 7. Testing a model and Evaluation with TensorRT models.
- Please follow [docs/7_Testing_Evaluation_with_TensoRT.md](docs/7_Testing_Evaluation_with_TensoRT.md) and proceed with the instructions.

------------------------------------------------------------------------------------------




## 3) Usage: Inference Method using ROS2 *Python* Node on the Container ENV

### 3.1 ROS2 play bagfile on the container
```
docker exec -it centerpointpillar bash
cd /Dataset
ros2 bag play segment-10359308928573410754_720_000_740_000_with_camera_labels/  # ros2 bag play folder_with_ros2bag
```

### 3.2 execute ros2_demo.py on the container
``` shell
docker exec -it centerpointpillar bash
cd ~/CenterPointPillar/tools/
python ros2_demo.py --cfg_file cfgs/waymo_models/centerpoint_pillar_inference.yaml --ckpt ../ckpt/checkpoint_epoch_24.pth
```

### 3.3 execute rviz2
``` shell
docker exec -it centerpointpillar bash
rviz2
```

### 3.4 setting rviz2
- Fixed Frame: base_link
- Add -> By display type -> PountCloud2 -> Topic: /lidar/top/pointcloud, Size(m): 0.03
- Add -> By topic -> /boxes/MarkerArray

<img src="sources/rviz2_add_topic.png" align="center" width="359">

### 3.5 run rviz2

<!-- show picture sources/fig1.png-->
<img src="sources/rviz2.png" align="center" width="100%">
<img src="sources/fig1.png" align="center" width="100%">

## 4) Usage: Inference Method using ROS2 *C++* Node on the Container ENV (Comming soon....)


### 4.3 ROS2 C++ Node
- Build the ROS2 package in your ROS2 workspace.
``` shell
cd ~/ && mkdir -p ros2_ws/src && cd ros2_ws/ && colcon build --symlink-install
cd src && ln -s CenterPointPillar/centerpoint .
cd src/centerpoint && mkdir model
cd ~/ros2_ws && colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```

### 4.4 Run the ROS2 Node.
``` shell
ros2 launch centerpoint centerpoint.launch.py
```

- Once running ros2 centerpoint node, create tensorRT file to the same folder having onnx file, automatically.

### 4.5 ROS2 play bagfile on the container
```
docker exec -it centerpointpillar bash
cd /Dataset
ros2 bag play segment-10359308928573410754_720_000_740_000_with_camera_labels/  # ros2 bag play folder_with_ros2bag
```

### 4.6 Run rviz2
``` shell
docker exec -it centerpointpillar bash
rviz2
```
- Fixed Frame: base_link
- Add -> By display type -> PountCloud2 -> Topic: /lidar/top/pointcloud, Size(m): 0.03
- Add -> By topic -> /boxes/MarkerArray

<img src="sources/rviz2_ros_cpp.png" align="center" width="100%">



