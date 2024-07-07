FROM nvcr.io/nvidia/tensorrt:23.04-py3

# Set environment variables
ENV NVENCODE_CFLAGS "-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal bash-completion sudo

# OpenCV
WORKDIR /opencv
RUN git clone https://github.com/opencv/opencv.git -b $CV_VERSION

WORKDIR /opencv/opencv/build

RUN cmake .. &&\
make -j12 &&\
make install &&\
ldconfig &&\
rm -rf /opencv

WORKDIR /
ENV OpenCV_DIR=/usr/share/OpenCV

# PyTorch for CUDA 12.1
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX"

# OpenPCDet Dependencies
RUN apt remove python3-blinker -y
RUN pip install -U pip
RUN pip install numpy==1.23.0 llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d==0.16.0 mayavi av2 kornia==0.6.8 pyquaternion colored
RUN pip install spconv-cu120
RUN pip install opencv-python==4.2.0.34
RUN pip install onnx==1.16.0
RUN pip install onnxsim==0.4.36
RUN pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install waymo-open-dataset-tf-2-12-0

ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="all"

# ROS2 Foxy installation
RUN sudo apt install software-properties-common libyaml-cpp-dev -y && \
    sudo add-apt-repository universe && \
    sudo apt update && sudo apt install curl -y && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && \
    sudo apt install ros-foxy-desktop python3-argcomplete -y && \
    sudo apt install ros-dev-tools ros-foxy-rqt* ros-foxy-tf-transformations -y
RUN sudo pip install transforms3d

