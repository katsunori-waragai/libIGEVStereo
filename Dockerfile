FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update
RUN apt install sudo
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
COPY sample.sh ./

RUN cd /root ; git clone https://github.com/katsunori-waragai/disparity-view.git
RUN cd /root/disparity-view; python3 -m pip install .[dev]
WORKDIR /root/libIGEVStereo

RUN python3 -m pip install .[dev]
