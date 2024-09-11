#!/bin/bash
xhost +
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	-v ${GIT_ROOT}/libIGEVStreo/test_data:/root/libIGEVStreo/test_data \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix libigev:100
 
