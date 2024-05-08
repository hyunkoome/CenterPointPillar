# Guidance to use OpenPCDet with docker

You can either build the docker image through Dockerfile or pull the docker image from dockerhub. Please make sure nvidia-docker is corretly installed.

## Build Through Dockerfile
Build the docker base image.
```shell script
docker build -f docker/env.Dockerfile -t openpcdet-env docker/
```
## User Configuration
Get your GID, UID, and username.
``` shell
# GID
echo $(id -g)
# UID
echo $(id -u)
# username
echo $(id -un)
```
Fill in the `.env` file with the GID, UID, and username like below.
``` shell
HOST_UID=1000
HOST_GID=1000
HOST_USER=jr
```
## Container Start
Start the container.
``` shell
docker compose up --build -d
```

