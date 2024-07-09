# Guidance to use OpenPCDet with docker

You can either build the docker image through Dockerfile or pull the docker image from dockerhub. Please make sure nvidia-docker is corretly installed.

## Build Through Dockerfile
Build the docker base image.
```shell script
cd CenterPointPillar
docker build -f docker/env.Dockerfile -t openpcdet-centerpoint-env docker/
```

## Container Start
- If you want to change account name of container, Before create the container
  - please change the `HOST_USER` from `lidar` to `you want` in the `.env` file of `root dir` (CenterPointPillar)
  - Fill in the `.env` file with the GID, UID, and username like below.
``` shell
HOST_UID=1000
HOST_GID=1000
HOST_USER=lidar
```````
- Create the container.
``` shell
cd CenterPointPillar
docker compose up --build -d
```

- Execute the container
```
docker exec -it centerpointpillar bash
```

- Get your GID, UID, and username in the container env 
``` shell
# GID -> 1000
echo $(id -g)  
# UID -> 1000
echo $(id -u)
# username -> lidar
echo $(id -un)
```








