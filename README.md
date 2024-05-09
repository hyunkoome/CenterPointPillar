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
### 4. PCDET Installation
``` shell
cd ~/OpenPCDet
sudo python setup.py develop
```