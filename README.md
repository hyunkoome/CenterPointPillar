<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

This repository is dedicated solely to inferencing the CenterPoint-pointpillar model.


## Environment
### 1. Docker Environment
- Base Image: [`nvcr.io/nvidia/tensorrt:22.04-py3`](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_22-04)
- OS: Ubuntu 20.24
- CUDA: 11.6.2
- cuBLAS: 11.9.3.114
- cuDNN: 8.4.0.27
- TensorRT: 8.2.4.2
- python: 3.8
- Pytorch: 1.13.1
### 2. Docker Container Start
Please refer to the [docker/README.md](docker/README.md) for more details.
### 3. PCDET Installation
``` shell
cd ~/OpenPCDet
sudo python setup.py develop
```
### 4. TensorRT Build w/ Plugin
TensorRT Plugin (PillarScatter Plugin) is required to build the TensorRT engine.
``` shell
cd tools/tensorrt_plugin/
cmake -B Release -DCMAKE_BUILD_TYPE=Release && cmake --build Release
```
ONNX to TensorRT conversion is also required.
``` shell
cd ~/OpenPCDet/
trtexec --onnx=./onnx/model.onnx --saveEngine=./model.trt --plugins=./tools/tensorrt_plugin/Release/libppscatter.so --fp16 --verbose
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
