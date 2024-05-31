#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/dev/bus/usb:/dev/bus/usb" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/home/peter/media/ssd2/BlenderProc:/blenderproc" \
--name=blenderproc_v0 blenderproc
