FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# for depth anything
RUN apt-get update
RUN apt install sudo
RUN apt install -y zip
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt install -y libv4l-dev v4l-utils qv4l2
RUN apt install -y curl
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install zstd
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab


RUN cd /root ; mkdir libIGEVStereo
RUN cd /root/libIGEVStereo
WORKDIR /root/libIGEVStereo
RUN mkdir /root/libIGEVStereo/libigev_stereo/
RUN mkdir /root/libIGEVStereo/libigev_stereo/utils
RUN mkdir /root/libIGEVStereo/scripts/
RUN mkdir /root/libIGEVStereo/demo-imgs/
COPY libigev_stereo/*.py /root/libIGEVStereo/libigev_stereo/
COPY libigev_stereo/utils/*.py /root/libIGEVStereo/libigev_stereo/utils/
# COPY scripts/*.py /root/libIGEVStereo/scripts/
COPY *.py ./
COPY demo-imgs/ /root/libIGEVStereo/demo-imgs/
RUN python3 -m pip install gdown
RUN mkdir -p /root/libIGEVStereo/libigev_stereo/pretrained_models/sceneflow/; cd /root/libIGEVStereo/libigev_stereo/pretrained_models/sceneflow/ ; gdown --fuzzy https://drive.google.com/file/d/16e9NR_RfzFdYT5mPaGwpjccplCi82C2e/view?usp=drive_link
COPY pyproject.toml ./
COPY sample.sh ./

RUN cd /root ; git clone https://github.com/katsunori-waragai/disparity-view.git
RUN cd /root/disparity-view; python3 -m pip install .[dev]
WORKDIR /root/libIGEVStereo

# RUN python3 -m pip install .[dev]
# ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
