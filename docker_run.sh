#!/bin/bash
xhost +
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	-v ${GIT_ROOT}/IGEV-docker/test_data:/root/IGEV/test_data \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix igev:100
 
