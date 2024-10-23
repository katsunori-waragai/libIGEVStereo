#!/bin/bash
xhost +
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so:$LD_PRELOAD
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v ${GIT_ROOT}/libIGEVStereo/mounted_data:/root/libIGEVStereo/mounted_data \
	-v ${GIT_ROOT}/libIGEVStereo/model_zoo:/root/libIGEVStereo/model_zoo \
	-v /tmp/.X11-unix/:/tmp/.X11-unix libigev:100
 
