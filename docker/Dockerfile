ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER
ENV HOST_UID=${HOST_UID?-1000}
ENV GROUP_ID=${HOST_GID?-1000}
ENV USER_NAME=${HOST_USER?-vueron}

# ENV PYTHONPATH="${PYTHONPATH}:/OpenPCDet"
RUN groupadd -g $HOST_GID $HOST_USER && \
    useradd -rm -u $HOST_UID -g $HOST_USER $HOST_USER && \
    usermod -aG sudo $HOST_USER && \
    echo "$HOST_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ROS2 Foxy installation
RUN sudo apt update && \
    sudo apt install software-properties-common && \
    sudo add-apt-repository universe && \
    sudo apt update && sudo apt install curl -y && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && \
    sudo apt install ros-foxy-desktop python3-argcomplete -y && \
    sudo apt install ros-dev-tools -y

RUN echo "source /opt/ros/foxy/setup.bash" >> /home/$HOST_USER/.bashrc
