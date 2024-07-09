## 1) Set Environment

### 1.1 Install Docker Engine on Ubuntu
- Please refer to the [`docker.docs`](https://docs.docker.com/engine/install/ubuntu/) for more details.
- If you would like to know more details, please refer to:
  - [`install guide for nvidia container toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
  - [`nvidia container toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit?tab=readme-ov-file) 

- docker 설치 후 /var/run/docker.sock의 permission denied 발생하는 경우
``` shell
sudo chmod 666 /var/run/docker.sock
```

### 1.2 Clone this repository
``` shell
git clone https://github.com/hyunkoome/CenterPointPillar.git
```
### 1.3 Docker Container Start

- Build the docker base image
```shell script
cd CenterPointPillar
docker build -f docker/env.Dockerfile -t openpcdet-centerpoint-env docker/
```

- Create the container.
``` shell
docker compose up --build -d
```

- Execute the container
```
docker exec -it centerpointpillar bash
```

- Please refer to the [docker/README.md](docker/README.md) for more details.

## [Return to the main page.](../README.md)