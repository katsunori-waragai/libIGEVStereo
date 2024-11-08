FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update
RUN apt install sudo
RUN apt install -y git-lfs
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab


RUN cd /root ; mkdir libIGEVStereo
RUN cd /root/libIGEVStereo
WORKDIR /root/libIGEVStereo
RUN mkdir -p /root/libIGEVStereo/stereoigev/
RUN mkdir /root/libIGEVStereo/scripts/
RUN mkdir -p /root/libIGEVStereo/test/test-imgs/
COPY stereoigev/*.py /root/libIGEVStereo/stereoigev/
COPY *.py ./
COPY test/test-imgs/ /root/libIGEVStereo/test/test-imgs/
COPY test/*.py test/*.sh /root/libIGEVStereo/test/
RUN python3 -m pip install gdown
RUN mkdir -p /root/libIGEVStereo/stereoigev/models/ ; cd /root/libIGEVStereo/stereoigev/models/ ; gdown --fuzzy https://drive.google.com/file/d/16e9NR_RfzFdYT5mPaGwpjccplCi82C2e/view?usp=drive_link
COPY pyproject.toml Makefile ./
COPY *.sh *.json ./
WORKDIR /root/libIGEVStereo

RUN python3 -m pip install .[dev]
ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so:$LD_PRELOAD
